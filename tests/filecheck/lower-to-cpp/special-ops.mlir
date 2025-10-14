// RUN: cpp-translate -i %s | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%c : i1, %x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.select"(%c, %x, %y) : (i1, !transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "select_test", "function_type" = (i1, !transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()
  
  "func.func"() ({
  ^0(%lhs : !transfer.abs_value<[!transfer.integer, !transfer.integer]>, %rhs : !transfer.abs_value<[!transfer.integer, !transfer.integer]>):
    %lhs0 = "transfer.get"(%lhs) {index = 0} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %lhs1 = "transfer.get"(%lhs) {index = 1} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %rhs0 = "transfer.get"(%rhs) {index = 0} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %rhs1 = "transfer.get"(%rhs) {index = 1} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %res0 = "transfer.or"(%lhs0, %rhs0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %res1 = "transfer.and"(%lhs1, %rhs1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %r = "transfer.make"(%res0, %res1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
    "func.return"(%r) : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> ()
  }) {"sym_name" = "kb_and_test", "function_type" = (!transfer.abs_value<[!transfer.integer, !transfer.integer]>, !transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.abs_value<[!transfer.integer, !transfer.integer]>):
    %r = "transfer.get"(%x) {index = 0} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_test", "function_type" = (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.make"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
    "func.return"(%r) : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> ()
  }) {"sym_name" = "make_2_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>} : () -> ()

  // "func.func"() ({
  // ^0(%x : !transfer.integer, %y : !transfer.integer, %z : !transfer.integer):
  //   %r = "transfer.make"(%x, %y, %z) : (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer, !transfer.integer]>
  //   "func.return"(%r) : (!transfer.abs_value<[!transfer.integer, !transfer.integer, !transfer.integer]>) -> ()
  // }) {"sym_name" = "make_3_test", "function_type" = (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer, !transfer.integer]>} : () -> ()
}) : () -> ()

// CHECK:      const APInt select_test(int c,const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = c ? x : y ;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: std::vector<const APInt> kb_and_test(std::vector<const APInt> &lhs,std::vector<const APInt> &rhs){
// CHECK-NEXT: 	const APInt lhs0 = lhs[0];
// CHECK-NEXT: 	const APInt lhs1 = lhs[1];
// CHECK-NEXT: 	const APInt rhs0 = rhs[0];
// CHECK-NEXT: 	const APInt rhs1 = rhs[1];
// CHECK-NEXT: 	const APInt res0 = lhs0|rhs0;
// CHECK-NEXT: 	const APInt res1 = lhs1&rhs1;
// CHECK-NEXT: 	std::vector<const APInt> r = std::vector<const APInt>{res0,res1};
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt get_test(std::vector<const APInt> &x){
// CHECK-NEXT: 	const APInt r = x[0];
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: std::vector<const APInt> make_2_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	std::vector<const APInt> r = std::vector<const APInt>{x,y};
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
