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

// CHECK:      const APInt add_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x+y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt sub_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x-y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt mul_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x*y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt and_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x&y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt or_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x|y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt xor_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x^y;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt udiv_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r;
// CHECK-NEXT: 	if (y == 0) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), -1);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.udiv(y);
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt sdiv_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r;
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
// CHECK-NEXT: const APInt urem_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r;
// CHECK-NEXT: 	if (y == 0) {
// CHECK-NEXT: 		r = x;
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.urem(y);
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt srem_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r;
// CHECK-NEXT: 	if (y == 0) {
// CHECK-NEXT: 		r = x;
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.srem(y);
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt shl_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r;
// CHECK-NEXT: 	if (y.uge(y.getBitWidth())) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), 0);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.shl(y.getZExtValue());
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt ashr_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r;
// CHECK-NEXT: 	if (y.uge(y.getBitWidth()) && x.isSignBitSet()) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), -1);
// CHECK-NEXT: 	} else if (y.uge(y.getBitWidth()) && x.isSignBitClear()) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), 0);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.ashr(y.getZExtValue());
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt lshr_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r;
// CHECK-NEXT: 	if (y.uge(y.getBitWidth())) {
// CHECK-NEXT: 		r = APInt(x.getBitWidth(), 0);
// CHECK-NEXT: 	} else {
// CHECK-NEXT: 		r = x.lshr(y.getZExtValue());
// CHECK-NEXT: 	}
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt umin_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = A::APIntOps::umin(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt smin_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = A::APIntOps::smin(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt umax_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = A::APIntOps::umax(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt smax_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = A::APIntOps::smax(x,y);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt get_high_bits_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x.getHiBits(y.getZExtValue());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt get_low_bits_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x.getLoBits(y.getZExtValue());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt set_high_bits_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.setHighBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.setHighBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt set_low_bits_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.setLowBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.setLowBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt clear_high_bits_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.clearHighBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.clearHighBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt clear_low_bits_test(const APInt &x,const APInt &y){
// CHECK-NEXT: 	const APInt r = x;
// CHECK-NEXT: 	if (y.ule(y.getBitWidth()))
// CHECK-NEXT: 		r.clearLowBits(y.getZExtValue());
// CHECK-NEXT: 	else
// CHECK-NEXT: 		r.clearLowBits(y.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
