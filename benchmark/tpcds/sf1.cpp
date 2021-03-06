#include "benchmark_runner.hpp"
#include "compare_result.hpp"
#include "dsdgen.hpp"
#include "duckdb_benchmark_macro.hpp"

using namespace duckdb;
using namespace std;

#define SF 1

#define TPCDS_QUERY_BODY(QNR)                                                                                          \
	virtual void Load(DuckDBBenchmarkState *state) {                                                                   \
		if (!BenchmarkRunner::TryLoadDatabase(state->db, "tpcds")) {                                                   \
			tpcds::dbgen(SF, state->db);                                                                               \
			BenchmarkRunner::SaveDatabase(state->db, "tpcds");                                                         \
		}                                                                                                              \
	}                                                                                                                  \
	virtual string GetQuery() {                                                                                        \
		return tpcds::get_query(QNR);                                                                                  \
	}                                                                                                                  \
	virtual string VerifyResult(QueryResult *result) {                                                                 \
		if (!result->success) {                                                                                        \
			return result->error;                                                                                      \
		}                                                                                                              \
		return ""; /*return compare_csv(*result, tpch::get_answer(SF, QNR),                                            \
		              true);  */                                                                                       \
	}                                                                                                                  \
	virtual string BenchmarkInfo() {                                                                                   \
		return StringUtil::Format("TPC-DS Q%d SF%d: %s", QNR, SF, tpcds::get_query(QNR).c_str());                      \
	}

DUCKDB_BENCHMARK(DSQ001, "[tpcds-sf1]")
TPCDS_QUERY_BODY(1);
FINISH_BENCHMARK(DSQ001);

DUCKDB_BENCHMARK(DSQ002, "[tpcds-sf1]")
TPCDS_QUERY_BODY(2);
FINISH_BENCHMARK(DSQ002);

DUCKDB_BENCHMARK(DSQ003, "[tpcds-sf1]")
TPCDS_QUERY_BODY(3);
FINISH_BENCHMARK(DSQ003);

DUCKDB_BENCHMARK(DSQ004, "[tpcds-sf1]")
TPCDS_QUERY_BODY(4);
FINISH_BENCHMARK(DSQ004);

DUCKDB_BENCHMARK(DSQ005, "[tpcds-sf1]")
TPCDS_QUERY_BODY(5);
FINISH_BENCHMARK(DSQ005);

DUCKDB_BENCHMARK(DSQ006, "[tpcds-sf1]")
TPCDS_QUERY_BODY(6);
FINISH_BENCHMARK(DSQ006);

DUCKDB_BENCHMARK(DSQ007, "[tpcds-sf1]")
TPCDS_QUERY_BODY(7);
FINISH_BENCHMARK(DSQ007);

DUCKDB_BENCHMARK(DSQ008, "[tpcds-sf1]")
TPCDS_QUERY_BODY(8);
FINISH_BENCHMARK(DSQ008);

DUCKDB_BENCHMARK(DSQ009, "[tpcds-sf1]")
TPCDS_QUERY_BODY(9);
FINISH_BENCHMARK(DSQ009);

DUCKDB_BENCHMARK(DSQ010, "[tpcds-sf1]")
TPCDS_QUERY_BODY(10);
FINISH_BENCHMARK(DSQ010);

DUCKDB_BENCHMARK(DSQ011, "[tpcds-sf1]")
TPCDS_QUERY_BODY(11);
FINISH_BENCHMARK(DSQ011);

DUCKDB_BENCHMARK(DSQ012, "[tpcds-sf1]")
TPCDS_QUERY_BODY(12);
FINISH_BENCHMARK(DSQ012);

DUCKDB_BENCHMARK(DSQ013, "[tpcds-sf1]")
TPCDS_QUERY_BODY(13);
FINISH_BENCHMARK(DSQ013);

DUCKDB_BENCHMARK(DSQ014, "[tpcds-sf1]")
TPCDS_QUERY_BODY(14);
FINISH_BENCHMARK(DSQ014);

DUCKDB_BENCHMARK(DSQ015, "[tpcds-sf1]")
TPCDS_QUERY_BODY(15);
FINISH_BENCHMARK(DSQ015);

DUCKDB_BENCHMARK(DSQ016, "[tpcds-sf1]")
TPCDS_QUERY_BODY(16);
FINISH_BENCHMARK(DSQ016);

DUCKDB_BENCHMARK(DSQ017, "[tpcds-sf1]")
TPCDS_QUERY_BODY(17);
FINISH_BENCHMARK(DSQ017);

DUCKDB_BENCHMARK(DSQ018, "[tpcds-sf1]")
TPCDS_QUERY_BODY(18);
FINISH_BENCHMARK(DSQ018);

DUCKDB_BENCHMARK(DSQ019, "[tpcds-sf1]")
TPCDS_QUERY_BODY(19);
FINISH_BENCHMARK(DSQ019);

DUCKDB_BENCHMARK(DSQ020, "[tpcds-sf1]")
TPCDS_QUERY_BODY(20);
FINISH_BENCHMARK(DSQ020);

DUCKDB_BENCHMARK(DSQ021, "[tpcds-sf1]")
TPCDS_QUERY_BODY(21);
FINISH_BENCHMARK(DSQ021);

DUCKDB_BENCHMARK(DSQ022, "[tpcds-sf1]")
TPCDS_QUERY_BODY(22);
FINISH_BENCHMARK(DSQ022);

DUCKDB_BENCHMARK(DSQ023, "[tpcds-sf1]")
TPCDS_QUERY_BODY(23);
FINISH_BENCHMARK(DSQ023);

DUCKDB_BENCHMARK(DSQ024, "[tpcds-sf1]")
TPCDS_QUERY_BODY(24);
FINISH_BENCHMARK(DSQ024);

DUCKDB_BENCHMARK(DSQ025, "[tpcds-sf1]")
TPCDS_QUERY_BODY(25);
FINISH_BENCHMARK(DSQ025);

DUCKDB_BENCHMARK(DSQ026, "[tpcds-sf1]")
TPCDS_QUERY_BODY(26);
FINISH_BENCHMARK(DSQ026);

DUCKDB_BENCHMARK(DSQ027, "[tpcds-sf1]")
TPCDS_QUERY_BODY(27);
FINISH_BENCHMARK(DSQ027);

DUCKDB_BENCHMARK(DSQ028, "[tpcds-sf1]")
TPCDS_QUERY_BODY(28);
FINISH_BENCHMARK(DSQ028);

DUCKDB_BENCHMARK(DSQ029, "[tpcds-sf1]")
TPCDS_QUERY_BODY(29);
FINISH_BENCHMARK(DSQ029);

DUCKDB_BENCHMARK(DSQ030, "[tpcds-sf1]")
TPCDS_QUERY_BODY(30);
FINISH_BENCHMARK(DSQ030);

DUCKDB_BENCHMARK(DSQ031, "[tpcds-sf1]")
TPCDS_QUERY_BODY(31);
FINISH_BENCHMARK(DSQ031);

DUCKDB_BENCHMARK(DSQ032, "[tpcds-sf1]")
TPCDS_QUERY_BODY(32);
FINISH_BENCHMARK(DSQ032);

DUCKDB_BENCHMARK(DSQ033, "[tpcds-sf1]")
TPCDS_QUERY_BODY(33);
FINISH_BENCHMARK(DSQ033);

DUCKDB_BENCHMARK(DSQ034, "[tpcds-sf1]")
TPCDS_QUERY_BODY(34);
FINISH_BENCHMARK(DSQ034);

DUCKDB_BENCHMARK(DSQ035, "[tpcds-sf1]")
TPCDS_QUERY_BODY(35);
FINISH_BENCHMARK(DSQ035);

DUCKDB_BENCHMARK(DSQ036, "[tpcds-sf1]")
TPCDS_QUERY_BODY(36);
FINISH_BENCHMARK(DSQ036);

DUCKDB_BENCHMARK(DSQ037, "[tpcds-sf1]")
TPCDS_QUERY_BODY(37);
FINISH_BENCHMARK(DSQ037);

DUCKDB_BENCHMARK(DSQ038, "[tpcds-sf1]")
TPCDS_QUERY_BODY(38);
FINISH_BENCHMARK(DSQ038);

DUCKDB_BENCHMARK(DSQ039, "[tpcds-sf1]")
TPCDS_QUERY_BODY(39);
FINISH_BENCHMARK(DSQ039);

DUCKDB_BENCHMARK(DSQ040, "[tpcds-sf1]")
TPCDS_QUERY_BODY(40);
FINISH_BENCHMARK(DSQ040);

DUCKDB_BENCHMARK(DSQ041, "[tpcds-sf1]")
TPCDS_QUERY_BODY(41);
FINISH_BENCHMARK(DSQ041);

DUCKDB_BENCHMARK(DSQ042, "[tpcds-sf1]")
TPCDS_QUERY_BODY(42);
FINISH_BENCHMARK(DSQ042);

DUCKDB_BENCHMARK(DSQ043, "[tpcds-sf1]")
TPCDS_QUERY_BODY(43);
FINISH_BENCHMARK(DSQ043);

DUCKDB_BENCHMARK(DSQ044, "[tpcds-sf1]")
TPCDS_QUERY_BODY(44);
FINISH_BENCHMARK(DSQ044);

DUCKDB_BENCHMARK(DSQ045, "[tpcds-sf1]")
TPCDS_QUERY_BODY(45);
FINISH_BENCHMARK(DSQ045);

DUCKDB_BENCHMARK(DSQ046, "[tpcds-sf1]")
TPCDS_QUERY_BODY(46);
FINISH_BENCHMARK(DSQ046);

DUCKDB_BENCHMARK(DSQ047, "[tpcds-sf1]")
TPCDS_QUERY_BODY(47);
FINISH_BENCHMARK(DSQ047);

DUCKDB_BENCHMARK(DSQ048, "[tpcds-sf1]")
TPCDS_QUERY_BODY(48);
FINISH_BENCHMARK(DSQ048);

DUCKDB_BENCHMARK(DSQ049, "[tpcds-sf1]")
TPCDS_QUERY_BODY(49);
FINISH_BENCHMARK(DSQ049);

DUCKDB_BENCHMARK(DSQ050, "[tpcds-sf1]")
TPCDS_QUERY_BODY(50);
FINISH_BENCHMARK(DSQ050);

DUCKDB_BENCHMARK(DSQ051, "[tpcds-sf1]")
TPCDS_QUERY_BODY(51);
FINISH_BENCHMARK(DSQ051);

DUCKDB_BENCHMARK(DSQ052, "[tpcds-sf1]")
TPCDS_QUERY_BODY(52);
FINISH_BENCHMARK(DSQ052);

DUCKDB_BENCHMARK(DSQ053, "[tpcds-sf1]")
TPCDS_QUERY_BODY(53);
FINISH_BENCHMARK(DSQ053);

DUCKDB_BENCHMARK(DSQ054, "[tpcds-sf1]")
TPCDS_QUERY_BODY(54);
FINISH_BENCHMARK(DSQ054);

DUCKDB_BENCHMARK(DSQ055, "[tpcds-sf1]")
TPCDS_QUERY_BODY(55);
FINISH_BENCHMARK(DSQ055);

DUCKDB_BENCHMARK(DSQ056, "[tpcds-sf1]")
TPCDS_QUERY_BODY(56);
FINISH_BENCHMARK(DSQ056);

DUCKDB_BENCHMARK(DSQ057, "[tpcds-sf1]")
TPCDS_QUERY_BODY(57);
FINISH_BENCHMARK(DSQ057);

DUCKDB_BENCHMARK(DSQ058, "[tpcds-sf1]")
TPCDS_QUERY_BODY(58);
FINISH_BENCHMARK(DSQ058);

DUCKDB_BENCHMARK(DSQ059, "[tpcds-sf1]")
TPCDS_QUERY_BODY(59);
FINISH_BENCHMARK(DSQ059);

DUCKDB_BENCHMARK(DSQ060, "[tpcds-sf1]")
TPCDS_QUERY_BODY(60);
FINISH_BENCHMARK(DSQ060);

DUCKDB_BENCHMARK(DSQ061, "[tpcds-sf1]")
TPCDS_QUERY_BODY(61);
FINISH_BENCHMARK(DSQ061);

DUCKDB_BENCHMARK(DSQ062, "[tpcds-sf1]")
TPCDS_QUERY_BODY(62);
FINISH_BENCHMARK(DSQ062);

DUCKDB_BENCHMARK(DSQ063, "[tpcds-sf1]")
TPCDS_QUERY_BODY(63);
FINISH_BENCHMARK(DSQ063);

DUCKDB_BENCHMARK(DSQ064, "[tpcds-sf1]")
TPCDS_QUERY_BODY(64);
FINISH_BENCHMARK(DSQ064);

DUCKDB_BENCHMARK(DSQ065, "[tpcds-sf1]")
TPCDS_QUERY_BODY(65);
FINISH_BENCHMARK(DSQ065);

DUCKDB_BENCHMARK(DSQ066, "[tpcds-sf1]")
TPCDS_QUERY_BODY(66);
FINISH_BENCHMARK(DSQ066);

DUCKDB_BENCHMARK(DSQ067, "[tpcds-sf1]")
TPCDS_QUERY_BODY(67);
FINISH_BENCHMARK(DSQ067);

DUCKDB_BENCHMARK(DSQ068, "[tpcds-sf1]")
TPCDS_QUERY_BODY(68);
FINISH_BENCHMARK(DSQ068);

DUCKDB_BENCHMARK(DSQ069, "[tpcds-sf1]")
TPCDS_QUERY_BODY(69);
FINISH_BENCHMARK(DSQ069);

DUCKDB_BENCHMARK(DSQ070, "[tpcds-sf1]")
TPCDS_QUERY_BODY(70);
FINISH_BENCHMARK(DSQ070);

DUCKDB_BENCHMARK(DSQ071, "[tpcds-sf1]")
TPCDS_QUERY_BODY(71);
FINISH_BENCHMARK(DSQ071);

DUCKDB_BENCHMARK(DSQ072, "[tpcds-sf1]")
TPCDS_QUERY_BODY(72);
FINISH_BENCHMARK(DSQ072);

DUCKDB_BENCHMARK(DSQ073, "[tpcds-sf1]")
TPCDS_QUERY_BODY(73);
FINISH_BENCHMARK(DSQ073);

DUCKDB_BENCHMARK(DSQ074, "[tpcds-sf1]")
TPCDS_QUERY_BODY(74);
FINISH_BENCHMARK(DSQ074);

DUCKDB_BENCHMARK(DSQ075, "[tpcds-sf1]")
TPCDS_QUERY_BODY(75);
FINISH_BENCHMARK(DSQ075);

DUCKDB_BENCHMARK(DSQ076, "[tpcds-sf1]")
TPCDS_QUERY_BODY(76);
FINISH_BENCHMARK(DSQ076);

DUCKDB_BENCHMARK(DSQ077, "[tpcds-sf1]")
TPCDS_QUERY_BODY(77);
FINISH_BENCHMARK(DSQ077);

DUCKDB_BENCHMARK(DSQ078, "[tpcds-sf1]")
TPCDS_QUERY_BODY(78);
FINISH_BENCHMARK(DSQ078);

DUCKDB_BENCHMARK(DSQ079, "[tpcds-sf1]")
TPCDS_QUERY_BODY(79);
FINISH_BENCHMARK(DSQ079);

DUCKDB_BENCHMARK(DSQ080, "[tpcds-sf1]")
TPCDS_QUERY_BODY(80);
FINISH_BENCHMARK(DSQ080);

DUCKDB_BENCHMARK(DSQ081, "[tpcds-sf1]")
TPCDS_QUERY_BODY(81);
FINISH_BENCHMARK(DSQ081);

DUCKDB_BENCHMARK(DSQ082, "[tpcds-sf1]")
TPCDS_QUERY_BODY(82);
FINISH_BENCHMARK(DSQ082);

DUCKDB_BENCHMARK(DSQ083, "[tpcds-sf1]")
TPCDS_QUERY_BODY(83);
FINISH_BENCHMARK(DSQ083);

DUCKDB_BENCHMARK(DSQ084, "[tpcds-sf1]")
TPCDS_QUERY_BODY(84);
FINISH_BENCHMARK(DSQ084);

DUCKDB_BENCHMARK(DSQ085, "[tpcds-sf1]")
TPCDS_QUERY_BODY(85);
FINISH_BENCHMARK(DSQ085);

DUCKDB_BENCHMARK(DSQ086, "[tpcds-sf1]")
TPCDS_QUERY_BODY(86);
FINISH_BENCHMARK(DSQ086);

DUCKDB_BENCHMARK(DSQ087, "[tpcds-sf1]")
TPCDS_QUERY_BODY(87);
FINISH_BENCHMARK(DSQ087);

DUCKDB_BENCHMARK(DSQ088, "[tpcds-sf1]")
TPCDS_QUERY_BODY(88);
FINISH_BENCHMARK(DSQ088);

DUCKDB_BENCHMARK(DSQ089, "[tpcds-sf1]")
TPCDS_QUERY_BODY(89);
FINISH_BENCHMARK(DSQ089);

DUCKDB_BENCHMARK(DSQ090, "[tpcds-sf1]")
TPCDS_QUERY_BODY(90);
FINISH_BENCHMARK(DSQ090);

DUCKDB_BENCHMARK(DSQ091, "[tpcds-sf1]")
TPCDS_QUERY_BODY(91);
FINISH_BENCHMARK(DSQ091);

DUCKDB_BENCHMARK(DSQ092, "[tpcds-sf1]")
TPCDS_QUERY_BODY(92);
FINISH_BENCHMARK(DSQ092);

DUCKDB_BENCHMARK(DSQ093, "[tpcds-sf1]")
TPCDS_QUERY_BODY(93);
FINISH_BENCHMARK(DSQ093);

DUCKDB_BENCHMARK(DSQ094, "[tpcds-sf1]")
TPCDS_QUERY_BODY(94);
FINISH_BENCHMARK(DSQ094);

DUCKDB_BENCHMARK(DSQ095, "[tpcds-sf1]")
TPCDS_QUERY_BODY(95);
FINISH_BENCHMARK(DSQ095);

DUCKDB_BENCHMARK(DSQ096, "[tpcds-sf1]")
TPCDS_QUERY_BODY(96);
FINISH_BENCHMARK(DSQ096);

DUCKDB_BENCHMARK(DSQ097, "[tpcds-sf1]")
TPCDS_QUERY_BODY(97);
FINISH_BENCHMARK(DSQ097);

DUCKDB_BENCHMARK(DSQ098, "[tpcds-sf1]")
TPCDS_QUERY_BODY(98);
FINISH_BENCHMARK(DSQ098);

DUCKDB_BENCHMARK(DSQ099, "[tpcds-sf1]")
TPCDS_QUERY_BODY(99);
FINISH_BENCHMARK(DSQ099);

DUCKDB_BENCHMARK(DSQ100, "[tpcds-sf1]")
TPCDS_QUERY_BODY(100);
FINISH_BENCHMARK(DSQ100);

DUCKDB_BENCHMARK(DSQ101, "[tpcds-sf1]")
TPCDS_QUERY_BODY(101);
FINISH_BENCHMARK(DSQ101);

DUCKDB_BENCHMARK(DSQ102, "[tpcds-sf1]")
TPCDS_QUERY_BODY(102);
FINISH_BENCHMARK(DSQ102);

DUCKDB_BENCHMARK(DSQ103, "[tpcds-sf1]")
TPCDS_QUERY_BODY(103);
FINISH_BENCHMARK(DSQ103);
