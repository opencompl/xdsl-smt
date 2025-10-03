// RUN: cpp-translate -i %s | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.umul_overflow"(%x, %y) : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "umul_ov_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.smul_overflow"(%x, %y) : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "smul_ov_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.uadd_overflow"(%x, %y) : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "uadd_ov_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sadd_overflow"(%x, %y) : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "sadd_ov_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.ushl_overflow"(%x, %y) : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "ushl_ov_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sshl_overflow"(%x, %y) : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "sshl_ov_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.intersects"(%x, %y) : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "intersects_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 0} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "eq_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 1} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "neq_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 2} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "slt_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 3} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "sle_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 4} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "sgt_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 5} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "sge_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 6} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "ult_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 7} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "ule_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 8} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "ugt_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.cmp"(%x, %y) {predicate = 9} : (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "uge_test", "function_type" = (!transfer.integer, !transfer.integer) -> i1} : () -> ()
}) : () -> ()

// CHECK:       int umul_ov_test(APInt x,APInt y){
// CHECK-NEXT:  	bool r;
// CHECK-NEXT:  	x.umul_ov(y,r);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int smul_ov_test(APInt x,APInt y){
// CHECK-NEXT:  	bool r;
// CHECK-NEXT:  	x.smul_ov(y,r);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int uadd_ov_test(APInt x,APInt y){
// CHECK-NEXT:  	bool r;
// CHECK-NEXT:  	x.uadd_ov(y,r);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int sadd_ov_test(APInt x,APInt y){
// CHECK-NEXT:  	bool r;
// CHECK-NEXT:  	x.sadd_ov(y,r);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int ushl_ov_test(APInt x,APInt y){
// CHECK-NEXT:  	bool r;
// CHECK-NEXT:  	x.ushl_ov(y,r);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int sshl_ov_test(APInt x,APInt y){
// CHECK-NEXT:  	bool r;
// CHECK-NEXT:  	x.sshl_ov(y,r);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int intersects_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.intersects(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int eq_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.eq(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int neq_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.ne(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int slt_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.slt(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int sle_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.sle(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int sgt_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.sgt(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int sge_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.sge(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int ult_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.ult(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int ule_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.ule(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int ugt_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.ugt(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int uge_test(APInt x,APInt y){
// CHECK-NEXT:  	int r = x.uge(y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
