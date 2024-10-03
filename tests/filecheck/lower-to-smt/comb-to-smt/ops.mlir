// RUN: xdsl-smt "%s" -p=lower-to-smt,lower-effects,canonicalize-smt -t=smt --split-input-file | filecheck "%s"

// comb.add
builtin.module {
  "func.func"() ({
  ^0(%x: i32):
    %r = "comb.add"(%x) : (i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_one", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:  (define-fun add_one ((x (Pair (_ BitVec 32) Bool)) (tmp Bool)) (Pair (Pair (_ BitVec 32) Bool) Bool)
// CHECK-NEXT:    (pair (pair (first x) (second x)) tmp))


  "func.func"() ({
  ^0(%x: i32, %y: i32):
    %r = "comb.add"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_two", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun add_two ((x_0 (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool)) (tmp_0 Bool)) (Pair (Pair (_ BitVec 32) Bool) Bool)
// CHECK-NEXT:    (pair (pair (bvadd (first x_0) (first y)) (or (second x_0) (second y))) tmp_0))

  "func.func"() ({
  ^0(%x: i32, %y: i32, %z: i32):
    %r = "comb.add"(%x, %y, %z) : (i32, i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_three", "function_type" = (i32, i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun add_three ((x_1 (Pair (_ BitVec 32) Bool)) (y_0 (Pair (_ BitVec 32) Bool)) (z (Pair (_ BitVec 32) Bool)) (tmp_1 Bool)) (Pair (Pair (_ BitVec 32) Bool) Bool)
// CHECK-NEXT:    (pair (pair (bvadd (bvadd (first x_1) (first y_0)) (first z)) (or (or (second x_1) (second y_0)) (second z))) tmp_1))

}

// -----

// comb.mul

builtin.module {
  "func.func"() ({
  ^0(%x: i32):
    %r = "comb.mul"(%x) : (i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "mul_one", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:  (define-fun mul_one ((x (Pair (_ BitVec 32) Bool)) (tmp Bool)) (Pair (Pair (_ BitVec 32) Bool) Bool)
// CHECK-NEXT:    (pair (pair (first x) (second x)) tmp))


  "func.func"() ({
  ^0(%x: i32, %y: i32):
    %r = "comb.mul"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "mul_two", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun mul_two ((x_0 (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool)) (tmp_0 Bool)) (Pair (Pair (_ BitVec 32) Bool) Bool)
// CHECK-NEXT:    (pair (pair (bvmul (first x_0) (first y)) (or (second x_0) (second y))) tmp_0))

  "func.func"() ({
  ^0(%x: i32, %y: i32, %z: i32):
    %r = "comb.mul"(%x, %y, %z) : (i32, i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "mul_three", "function_type" = (i32, i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun mul_three ((x_1 (Pair (_ BitVec 32) Bool)) (y_0 (Pair (_ BitVec 32) Bool)) (z (Pair (_ BitVec 32) Bool)) (tmp_1 Bool)) (Pair (Pair (_ BitVec 32) Bool) Bool)
// CHECK-NEXT:    (pair (pair (bvmul (bvmul (first x_1) (first y_0)) (first z)) (or (or (second x_1) (second y_0)) (second z))) tmp_1))

}
