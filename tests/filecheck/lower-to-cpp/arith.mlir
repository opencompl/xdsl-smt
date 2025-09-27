// RUN: cpp-translate -i %s | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.addi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_test", "function_type" = (i32, i32) -> i32} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.subi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "sub_test", "function_type" = (i32, i32) -> i32} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.andi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "and_test", "function_type" = (i32, i32) -> i32} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.ori"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "or_test", "function_type" = (i32, i32) -> i32} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.xori"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "xor_test", "function_type" = (i32, i32) -> i32} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 0 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "eq_test", "function_type" = (i32, i32) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 1 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "neq_test", "function_type" = (i32, i32) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 2 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "lt_test", "function_type" = (i32, i32) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 3 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "leq_test", "function_type" = (i32, i32) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 4 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "gt_test", "function_type" = (i32, i32) -> i1} : () -> ()

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 5 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "geq_test", "function_type" = (i32, i32) -> i1} : () -> ()

  "func.func"() ({
    %x = "arith.constant"() {value = 3 : i32} : () -> i32
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "const_test", "function_type" = () -> i32} : () -> ()

  "func.func"() ({
  ^0(%x : i32):
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "empty_func_test", "function_type" = (i32) -> i32} : () -> ()
}) : () -> ()

// CHECK:       int add_test(int x,int y){
// CHECK-NEXT:  	int r = x+y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int sub_test(int x,int y){
// CHECK-NEXT:  	int r = x-y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int and_test(int x,int y){
// CHECK-NEXT:  	int r = x&y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int or_test(int x,int y){
// CHECK-NEXT:  	int r = x|y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int xor_test(int x,int y){
// CHECK-NEXT:  	int r = x^y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int eq_test(int x,int y){
// CHECK-NEXT:  	int r = (x==y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int neq_test(int x,int y){
// CHECK-NEXT:  	int r = (x!=y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int lt_test(int x,int y){
// CHECK-NEXT:  	int r = (x<y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int leq_test(int x,int y){
// CHECK-NEXT:  	int r = (x<=y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int gt_test(int x,int y){
// CHECK-NEXT:  	int r = (x>y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int geq_test(int x,int y){
// CHECK-NEXT:  	int r = (x>=y);
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  int const_test(){
// CHECK-NEXT:  	int x = 3;
// CHECK-NEXT:  	return x;
// CHECK-NEXT:  }
// CHECK-NEXT:  int empty_func_test(int x){
// CHECK-NEXT:  	return x;
// CHECK-NEXT:  }
