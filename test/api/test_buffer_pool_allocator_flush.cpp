#include "catch.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/storage/block_allocator.hpp"
#include "duckdb/storage/buffer/buffer_pool.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "test_helpers.hpp"

using namespace duckdb; // NOLINT

TEST_CASE("Buffer pool flushes the allocator after bulk deallocation", "[storage][buffer_pool]") {
	DBConfig config;
	config.options.maximum_memory = 256ULL * 1024 * 1024;
	// the block allocator pool supports flushing on every platform, unlike the fallback allocator
	config.options.block_allocator_size = 64ULL * 1024 * 1024;
	DuckDB db(nullptr, &config);
	auto &db_instance = *db.instance;
	if (!BlockAllocator::Get(db_instance).SupportsFlush()) {
		return;
	}
	auto &buffer_manager = BufferManager::GetBufferManager(db_instance);
	auto &pool = db_instance.GetBufferPool();
	const auto block_size = buffer_manager.GetBlockAllocSize();
	const auto blocks_per_mb = (1024 * 1024) / block_size;

	auto allocate_and_free = [&](idx_t mb) {
		vector<BufferHandle> handles;
		for (idx_t i = 0; i < mb * blocks_per_mb; i++) {
			handles.push_back(buffer_manager.Allocate(MemoryTag::EXTENSION, block_size));
		}
	};

	// with a 256MB limit, the pool flushes after 16MB has been deallocated
	const auto flushes = pool.GetAllocatorFlushCount();
	allocate_and_free(4);
	auto handle = buffer_manager.Allocate(MemoryTag::EXTENSION, block_size);
	REQUIRE(pool.GetAllocatorFlushCount() == flushes);

	allocate_and_free(20);
	REQUIRE(pool.GetAllocatorFlushCount() == flushes);
	// the flush happens on the next allocation, before memory grows again
	handle = buffer_manager.Allocate(MemoryTag::EXTENSION, block_size);
	REQUIRE(pool.GetAllocatorFlushCount() == flushes + 1);
	handle = buffer_manager.Allocate(MemoryTag::EXTENSION, block_size);
	REQUIRE(pool.GetAllocatorFlushCount() == flushes + 1);

	// a lower setting lowers the amount that has to be deallocated first
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET allocator_bulk_deallocation_flush_threshold='2MB'"));
	allocate_and_free(4);
	handle = buffer_manager.Allocate(MemoryTag::EXTENSION, block_size);
	REQUIRE(pool.GetAllocatorFlushCount() == flushes + 2);

	// a higher setting does not raise it above the fraction of the memory limit
	REQUIRE_NO_FAIL(con.Query("SET allocator_bulk_deallocation_flush_threshold='1GB'"));
	allocate_and_free(4);
	handle = buffer_manager.Allocate(MemoryTag::EXTENSION, block_size);
	REQUIRE(pool.GetAllocatorFlushCount() == flushes + 2);
	allocate_and_free(20);
	handle = buffer_manager.Allocate(MemoryTag::EXTENSION, block_size);
	REQUIRE(pool.GetAllocatorFlushCount() == flushes + 3);
}
