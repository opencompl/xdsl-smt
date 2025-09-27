// RUN: cpp-translate -i %s | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%c : i1, %x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.select"(%c, %x, %y) : (i1, !transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "select_test", "function_type" = (i1, !transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.make"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
    "func.return"(%r) : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> ()
  }) {"sym_name" = "make_2_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>} : () -> ()
  
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer, %z : !transfer.integer):
    %r = "transfer.make"(%x, %y, %z) : (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer, !transfer.integer]>
    "func.return"(%r) : (!transfer.abs_value<[!transfer.integer, !transfer.integer, !transfer.integer]>) -> ()
  }) {"sym_name" = "make_3_test", "function_type" = (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer, !transfer.integer]>} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.abs_value<[!transfer.integer, !transfer.integer]>):
    %r = "transfer.get"(%x) {index = 0} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_test", "function_type" = (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer} : () -> ()
}) : () -> ()

// CHECK:       APInt select_test(int c,APInt x,APInt y){
// CHECK-NEXT:  	APInt r = c ? x : y ;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  std::vector<APInt> make_2_test(APInt x,APInt y){
// CHECK-NEXT:  	std::vector<APInt> r = std::vector<APInt>{x,y};
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  std::vector<APInt> make_3_test(APInt x,APInt y,APInt z){
// CHECK-NEXT:  	std::vector<APInt> r = std::vector<APInt>{x,y,z};
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  APInt get_test(std::vector<APInt> x){
// CHECK-NEXT:  	APInt r = x[0];
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
