#include "form/optimization/diagnostics.hpp"
#include <gtest/gtest.h>
#include <sstream>
#include <algorithm>
using namespace form::diagnostics;
TEST(Diagnostics, DisabledAndReset) {
  enabled=false; reset();
  { Scope scope(Stage::cpu_reset); }
  EXPECT_EQ(calls[static_cast<size_t>(Stage::cpu_reset)].load(),0);
  enabled=true;
  { Scope scope(Stage::cpu_reset); }
  EXPECT_EQ(calls[static_cast<size_t>(Stage::cpu_reset)].load(),1);
  reset();
  EXPECT_EQ(calls[static_cast<size_t>(Stage::cpu_reset)].load(),0);
  EXPECT_EQ(nanoseconds[static_cast<size_t>(Stage::cpu_reset)].load(),0);
  enabled=false;
}
TEST(Diagnostics, NestedInclusiveAccountingAndCsv) {
  enabled=true; reset();
  { Scope outer(Stage::cpu_reset); { Scope inner(Stage::batch_reset); } }
  auto parent=static_cast<size_t>(Stage::cpu_reset), child=static_cast<size_t>(Stage::batch_reset);
  EXPECT_EQ(calls[parent].load(),1); EXPECT_EQ(calls[child].load(),1);
  EXPECT_GE(nanoseconds[parent].load(),nanoseconds[child].load());
  std::ostringstream header,row; writeHeader(header); writeRow(row);
  auto h=header.str(),r=row.str();
  EXPECT_EQ(std::count(h.begin(),h.end(),','),std::count(r.begin(),r.end(),','));
  EXPECT_NE(h.find("diag_cpu_reset_ms"),std::string::npos);
  enabled=false; reset();
}
TEST(Diagnostics, OptimizerSnapshotIsDelta) {
  enabled=true; reset();
  tick(Stage::lm_accepted);
  { OptimizerCall call(false,true,60);
    { Scope reset_scope(Stage::cpu_reset); }
    tick(Stage::lm_accepted); tick(Stage::lm_rejected);
  }
  ASSERT_EQ(optimizers.size(),1);
  EXPECT_FALSE(optimizers[0].gpu); EXPECT_TRUE(optimizers[0].fast);
  EXPECT_EQ(optimizers[0].dimension,60);
  EXPECT_EQ(optimizers[0].n[static_cast<size_t>(Stage::lm_accepted)],1);
  EXPECT_EQ(optimizers[0].n[static_cast<size_t>(Stage::lm_rejected)],1);
  EXPECT_GE(optimizers[0].wall_ms,optimizers[0].ns[static_cast<size_t>(Stage::cpu_reset)]*1e-6);
  enabled=false; reset();
}
