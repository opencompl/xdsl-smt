// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt --split-input-file | filecheck "%s"

// comb.add

builtin.module {
  "func.func"() ({
  ^0():
    %r = "comb.add"() : () -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_none", "function_type" = () -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:      (define-fun add_none () (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (_ bv0 32) false))


  "func.func"() ({
  ^0(%x: i32):
    %r = "comb.add"(%x) : (i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_one", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun add_one (({{.*}} (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (first {{.*}}) (second {{.*}})))


  "func.func"() ({
  ^0(%x: i32, %y: i32):
    %r = "comb.add"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_two", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun add_two (({{.*}} (Pair (_ BitVec 32) Bool)) ({{.*}} (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (bvadd (first {{.*}}) (first {{.*}})) (or (second {{.*}}) (second {{.*}}))))

  "func.func"() ({
  ^0(%x: i32, %y: i32, %z: i32):
    %r = "comb.add"(%x, %y, %z) : (i32, i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "add_three", "function_type" = (i32, i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun add_three (({{.*}} (Pair (_ BitVec 32) Bool)) ({{.*}} (Pair (_ BitVec 32) Bool)) ({{.*}} (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (bvadd (bvadd (first {{.*}}) (first {{.*}})) (first {{.*}})) (or (or (second {{.*}}) (second {{.*}})) (second {{.*}}))))

}

// -----

// comb.mul

builtin.module {
  "func.func"() ({
  ^0():
    %r = "comb.mul"() : () -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "mul_none", "function_type" = () -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:  (define-fun mul_none () (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (_ bv1 32) false))

  "func.func"() ({
  ^0(%x: i32):
    %r = "comb.mul"(%x) : (i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "mul_one", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun mul_one (({{.*}} (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (first {{.*}}) (second {{.*}})))


  "func.func"() ({
  ^0(%x: i32, %y: i32):
    %r = "comb.mul"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "mul_two", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun mul_two (({{.*}} (Pair (_ BitVec 32) Bool)) ({{.*}} (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (bvmul (first {{.*}}) (first {{.*}})) (or (second {{.*}}) (second {{.*}}))))

  "func.func"() ({
  ^0(%x: i32, %y: i32, %z: i32):
    %r = "comb.mul"(%x, %y, %z) : (i32, i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "mul_three", "function_type" = (i32, i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK-NEXT:  (define-fun mul_three (({{.*}} (Pair (_ BitVec 32) Bool)) ({{.*}} (Pair (_ BitVec 32) Bool)) ({{.*}} (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (bvmul (bvmul (first {{.*}}) (first {{.*}})) (first {{.*}})) (or (or (second {{.*}}) (second {{.*}})) (second {{.*}}))))

}
