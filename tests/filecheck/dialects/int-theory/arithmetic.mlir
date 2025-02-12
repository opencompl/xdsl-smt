// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({
      %x = "smt.int.constant"() {value = 42} : () -> !smt.int.int
      %y = "smt.int.constant"() {value = 84} : () -> !smt.int.int

      %lt = "smt.int.lt"(%x, %y) : (!smt.int.int, !smt.int.int) -> !smt.bool
      %le = "smt.int.le"(%x, %y) : (!smt.int.int, !smt.int.int) -> !smt.bool
      "smt.assert"(%lt) : (!smt.bool) -> ()
      "smt.assert"(%le) : (!smt.bool) -> ()

      %gt = "smt.int.gt"(%y, %x) : (!smt.int.int, !smt.int.int) -> !smt.bool
      %ge = "smt.int.ge"(%y, %x) : (!smt.int.int, !smt.int.int) -> !smt.bool
      "smt.assert"(%gt) : (!smt.bool) -> ()
      "smt.assert"(%ge) : (!smt.bool) -> ()

      %nx = "smt.int.neg"(%x) : (!smt.int.int) -> !smt.int.int
      %ax = "smt.int.abs"(%nx) : (!smt.int.int) -> !smt.int.int
      %eq_neg_abs = "smt.eq"(%x, %ax) : (!smt.int.int, !smt.int.int) -> !smt.bool
      "smt.assert"(%eq_neg_abs) : (!smt.bool) -> ()

      %add = "smt.int.add"(%x, %y) : (!smt.int.int, !smt.int.int) -> !smt.int.int
      %sub = "smt.int.sub"(%add, %y) : (!smt.int.int, !smt.int.int) -> !smt.int.int
      %eq_add_sub = "smt.eq"(%x, %sub) : (!smt.int.int, !smt.int.int) -> !smt.bool
      "smt.assert"(%eq_add_sub) : (!smt.bool) -> ()

      %mul = "smt.int.mul"(%x, %y) : (!smt.int.int, !smt.int.int) -> !smt.int.int
      %div = "smt.int.div"(%mul, %y) : (!smt.int.int, !smt.int.int) -> !smt.int.int
      %eq_mul_div = "smt.eq"(%x, %div) : (!smt.int.int, !smt.int.int) -> !smt.bool
      "smt.assert"(%eq_mul_div) : (!smt.bool) -> ()

      %mod = "smt.int.mod"(%x, %y) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  }) : () -> ()

//CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT: (assert (let ((x 42))
// CHECK-NEXT:   (let ((y 84))
// CHECK-NEXT:   (< x y))))
// CHECK-NEXT: (assert (let ((x 42))
// CHECK-NEXT:   (let ((y 84))
// CHECK-NEXT:   (<= x y))))
// CHECK-NEXT: (assert (let ((y 84))
// CHECK-NEXT:   (let ((x 42))
// CHECK-NEXT:   (> y x))))
// CHECK-NEXT: (assert (let ((y 84))
// CHECK-NEXT:   (let ((x 42))
// CHECK-NEXT:   (>= y x))))
// CHECK-NEXT: (assert (let ((x 42))
// CHECK-NEXT:   (= x (abs (- x)))))
// CHECK-NEXT: (assert (let ((x 42))
// CHECK-NEXT:   (let ((y 84))
// CHECK-NEXT:   (= x (- (+ x y) y)))))
// CHECK-NEXT: (assert (let ((x 42))
// CHECK-NEXT:   (let ((y 84))
// CHECK-NEXT:   (= x (div (* x y) y)))))
