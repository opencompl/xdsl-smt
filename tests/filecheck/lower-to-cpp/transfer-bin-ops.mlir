// RUN: cpp-translate -i %s | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.add"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "add_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sub"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "sub_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.mul"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "mul_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.and"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "and_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.or"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "or_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.xor"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "xor_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.udiv"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "udiv_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sdiv"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "sdiv_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.urem"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "urem_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.srem"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "srem_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.shl"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "shl_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.ashr"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "ashr_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.lshr"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "lshr_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.umin"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "umin_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.smin"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "smin_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.umax"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "umax_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.smax"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "smax_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

 "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.get_high_bits"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_high_bits_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.get_low_bits"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_low_bits_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.set_high_bits"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "set_high_bits_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.set_low_bits"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "set_low_bits_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.clear_high_bits"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "clear_high_bits_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.clear_low_bits"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "clear_low_bits_test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer} : () -> ()
}) : () -> ()

// CHECK:       APInt add_test(APInt x,APInt y){
// CHECK-NEXT:  	APInt r = x+y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  APInt sub_test(APInt x,APInt y){
// CHECK-NEXT:  	APInt r = x-y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  APInt mul_test(APInt x,APInt y){
// CHECK-NEXT:  	APInt r = x*y;
// CHECK-NEXT:  	return r;
// CHECK-NEXT:  }
// CHECK-NEXT:  APInt and_test(APInt x,APInt y){
// CHECK-NEXT:  	APInt r = x&y;
// CHECK-NEXT:    return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt or_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x|y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt xor_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x^y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt udiv_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r;
// CHECK-NEXT: 	if (y == 0) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), -1);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.udiv(y);
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt sdiv_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r;
// CHECK-NEXT: 	if (x.isMinSignedValue() && y == -1) {
// CHECK-NEXT: 		r = APInt::getSignedMinValue(x.getBitWidth());
// CHECK-NEXT: 	} else if (y == 0 && x.isNonNegative()) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), -1);
// CHECK-NEXT: 	} else if (y == 0 && x.isNegative()) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), 1);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.sdiv(y);
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt urem_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r;
// CHECK-NEXT: 	if (y == 0) {
// CHECK-NEXT: 		r = x;
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.urem(y);
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt srem_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r;
// CHECK-NEXT: 	if (y == 0) {
// CHECK-NEXT: 		r = x;
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.srem(y);
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt shl_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r;
// CHECK-NEXT: 	if (y.uge(y.getBitWidth())) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), 0);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.shl(y.getZExtValue());
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt ashr_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r;
// CHECK-NEXT: 	if (y.uge(y.getBitWidth()) && x.isSignBitSet()) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), -1);
// CHECK-NEXT: 	} else if (y.uge(y.getBitWidth()) && x.isSignBitClear()) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), 0);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.ashr(y.getZExtValue());
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt lshr_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r;
// CHECK-NEXT: 	if (y.uge(y.getBitWidth())) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), 0);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.lshr(y.getZExtValue());
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt umin_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = A::APIntOps::umin(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt smin_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = A::APIntOps::smin(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt umax_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = A::APIntOps::umax(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt smax_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = A::APIntOps::smax(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt get_high_bits_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x.getHiBits(y.getZExtValue());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt get_low_bits_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x.getLoBits(y.getZExtValue());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt set_high_bits_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.setHighBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.setHighBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt set_low_bits_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.setLowBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.setLowBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt clear_high_bits_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.clearHighBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.clearHighBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: APInt clear_low_bits_test(APInt x,APInt y){
// CHECK-NEXT: 	APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.clearLowBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.clearLowBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
