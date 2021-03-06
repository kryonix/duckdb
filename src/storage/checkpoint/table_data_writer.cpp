#include "duckdb/storage/checkpoint/table_data_writer.hpp"

#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/types/null_value.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/serializer/buffered_serializer.hpp"

#include "duckdb/storage/numeric_segment.hpp"
#include "duckdb/storage/string_segment.hpp"
#include "duckdb/storage/table/column_segment.hpp"

using namespace duckdb;
using namespace std;

class WriteOverflowStringsToDisk : public OverflowStringWriter {
public:
	WriteOverflowStringsToDisk(CheckpointManager &manager);
	~WriteOverflowStringsToDisk();

	//! The checkpoint manager
	CheckpointManager &manager;
	//! Block handle use for writing to
	unique_ptr<BufferHandle> handle;
	//! The current block we are writing to
	block_id_t block_id;
	//! The offset within the current block
	index_t offset;

	static constexpr index_t STRING_SPACE = Storage::BLOCK_SIZE - sizeof(block_id_t);

public:
	void WriteString(string_t string, block_id_t &result_block, int32_t &result_offset) override;

private:
	void AllocateNewBlock(block_id_t new_block_id);
};

TableDataWriter::TableDataWriter(CheckpointManager &manager, TableCatalogEntry &table)
    : manager(manager), table(table) {
}

TableDataWriter::~TableDataWriter() {
}

void TableDataWriter::WriteTableData(Transaction &transaction) {
	// allocate segments to write the table to
	segments.resize(table.columns.size());
	data_pointers.resize(table.columns.size());
	for (index_t i = 0; i < table.columns.size(); i++) {
		auto type_id = GetInternalType(table.columns[i].type);
		stats.push_back(make_unique<SegmentStatistics>(type_id, GetTypeIdSize(type_id)));
		CreateSegment(i);
	}

	// now start scanning the table and append the data to the uncompressed segments
	vector<column_t> column_ids;
	for (auto &column : table.columns) {
		column_ids.push_back(column.oid);
	}
	// initialize scan structures to prepare for the scan
	TableScanState state;
	table.storage->InitializeScan(transaction, state, column_ids);
	//! get all types of the table and initialize the chunk
	auto types = table.GetTypes();
	DataChunk chunk;
	chunk.Initialize(types);

	while (true) {
		chunk.Reset();
		// now scan the table to construct the blocks
		table.storage->Scan(transaction, chunk, state);
		if (chunk.size() == 0) {
			break;
		}
		// for each column, we append whatever we can fit into the block
		for (index_t i = 0; i < table.columns.size(); i++) {
			assert(chunk.data[i].type == GetInternalType(table.columns[i].type));
			AppendData(i, chunk.data[i]);
		}
	}
	// flush any remaining data and write the data pointers to disk
	for (index_t i = 0; i < table.columns.size(); i++) {
		FlushSegment(i);
	}
	WriteDataPointers();
}

void TableDataWriter::CreateSegment(index_t col_idx) {
	auto type_id = GetInternalType(table.columns[col_idx].type);
	if (type_id == TypeId::VARCHAR) {
		auto string_segment = make_unique<StringSegment>(manager.buffer_manager, 0);
		string_segment->overflow_writer = make_unique<WriteOverflowStringsToDisk>(manager);
		segments[col_idx] = move(string_segment);
	} else {
		segments[col_idx] = make_unique<NumericSegment>(manager.buffer_manager, type_id, 0);
	}
}

void TableDataWriter::AppendData(index_t col_idx, Vector &data) {
	index_t count = data.count;
	index_t offset = 0;
	while (offset < count) {
		index_t appended = segments[col_idx]->Append(*stats[col_idx], data, offset, count);
		if (appended == count) {
			// appended everything: finished
			return;
		}
		// the segment is full: flush it to disk
		FlushSegment(col_idx);

		// now create a new segment and continue appending
		CreateSegment(col_idx);
		offset += appended;
		count -= appended;
	}
}

void TableDataWriter::FlushSegment(index_t col_idx) {
	auto tuple_count = segments[col_idx]->tuple_count;
	if (tuple_count == 0) {
		return;
	}

	// get the buffer of the segment and pin it
	auto handle = manager.buffer_manager.Pin(segments[col_idx]->block_id);

	// get a free block id to write to
	auto block_id = manager.block_manager.GetFreeBlockId();

	// construct the data pointer, FIXME: add statistics as well
	DataPointer data_pointer;
	data_pointer.block_id = block_id;
	data_pointer.offset = 0;
	data_pointer.row_start = 0;
	if (data_pointers[col_idx].size() > 0) {
		auto &last_pointer = data_pointers[col_idx].back();
		data_pointer.row_start = last_pointer.row_start + last_pointer.tuple_count;
	}
	data_pointer.tuple_count = tuple_count;
	data_pointers[col_idx].push_back(data_pointer);
	// write the block to disk
	manager.block_manager.Write(*handle->node, block_id);
}

void TableDataWriter::WriteDataPointers() {
	for (index_t i = 0; i < data_pointers.size(); i++) {
		// get a reference to the data column
		auto &data_pointer_list = data_pointers[i];
		manager.tabledata_writer->Write<index_t>(data_pointer_list.size());
		// then write the data pointers themselves
		for (index_t k = 0; k < data_pointer_list.size(); k++) {
			auto &data_pointer = data_pointer_list[k];
			manager.tabledata_writer->Write<double>(data_pointer.min);
			manager.tabledata_writer->Write<double>(data_pointer.max);
			manager.tabledata_writer->Write<index_t>(data_pointer.row_start);
			manager.tabledata_writer->Write<index_t>(data_pointer.tuple_count);
			manager.tabledata_writer->Write<block_id_t>(data_pointer.block_id);
			manager.tabledata_writer->Write<uint32_t>(data_pointer.offset);
		}
	}
}

WriteOverflowStringsToDisk::WriteOverflowStringsToDisk(CheckpointManager &manager)
    : manager(manager), handle(nullptr), block_id(INVALID_BLOCK), offset(0) {
}

WriteOverflowStringsToDisk::~WriteOverflowStringsToDisk() {
	if (offset > 0) {
		manager.block_manager.Write(*handle->node, block_id);
	}
}

void WriteOverflowStringsToDisk::WriteString(string_t string, block_id_t &result_block, int32_t &result_offset) {
	if (!handle) {
		handle = manager.buffer_manager.Allocate(Storage::BLOCK_ALLOC_SIZE);
	}
	// first write the length of the string
	if (block_id == INVALID_BLOCK || offset + sizeof(uint32_t) >= STRING_SPACE) {
		AllocateNewBlock(manager.block_manager.GetFreeBlockId());
	}
	result_block = block_id;
	result_offset = offset;

	// write the length field
	*((uint32_t *)(handle->node->buffer + offset)) = string.length;
	offset += sizeof(uint32_t);
	// now write the remainder of the string
	auto strptr = string.data;
	uint32_t remaining = string.length + 1;
	while (remaining > 0) {
		uint32_t to_write = std::min((uint32_t)remaining, (uint32_t)(STRING_SPACE - offset));
		if (to_write > 0) {
			memcpy(handle->node->buffer + offset, strptr, to_write);

			remaining -= to_write;
			offset += to_write;
			strptr += to_write;
		}
		if (remaining > 0) {
			// there is still remaining stuff to write
			// first get the new block id and write it to the end of the previous block
			auto new_block_id = manager.block_manager.GetFreeBlockId();
			*((block_id_t *)(handle->node->buffer + offset)) = new_block_id;
			// now write the current block to disk and allocate a new block
			AllocateNewBlock(new_block_id);
		}
	}
}

void WriteOverflowStringsToDisk::AllocateNewBlock(block_id_t new_block_id) {
	if (block_id != INVALID_BLOCK) {
		// there is an old block, write it first
		manager.block_manager.Write(*handle->node, block_id);
	}
	offset = 0;
	block_id = new_block_id;
}
