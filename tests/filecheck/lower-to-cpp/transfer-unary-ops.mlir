// RUN: cpp-translate -i %s | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.get_bit_width"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_bw_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.countl_zero"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "countl_zero_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.countr_zero"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "countr_zero_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.countl_one"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "countl_one_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.countr_one"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "countr_one_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.neg"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "neg_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.clear_sign_bit"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "clear_sign_bit_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.set_sign_bit"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "set_sign_bit_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.get_all_ones"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_all_ones_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.get_signed_max_value"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_signed_max_value_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.get_signed_min_value"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "get_signed_min_value_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.reverse_bits"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "reverse_bits_test", "function_type" = (!transfer.integer) -> !transfer.integer} : () -> ()
}) : () -> ()

// CHECK:      const APInt get_bw_test(const APInt &x){
// CHECK-NEXT: 	unsigned r_autocast = x.getBitWidth();
// CHECK-NEXT: 	APInt r(x.getBitWidth(),r_autocast);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt countl_zero_test(const APInt &x){
// CHECK-NEXT: 	unsigned r_autocast = x.countl_zero();
// CHECK-NEXT: 	APInt r(x.getBitWidth(),r_autocast);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt countr_zero_test(const APInt &x){
// CHECK-NEXT: 	unsigned r_autocast = x.countr_zero();
// CHECK-NEXT: 	APInt r(x.getBitWidth(),r_autocast);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt countl_one_test(const APInt &x){
// CHECK-NEXT: 	unsigned r_autocast = x.countl_one();
// CHECK-NEXT: 	APInt r(x.getBitWidth(),r_autocast);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt countr_one_test(const APInt &x){
// CHECK-NEXT: 	unsigned r_autocast = x.countr_one();
// CHECK-NEXT: 	APInt r(x.getBitWidth(),r_autocast);
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt neg_test(const APInt &x){
// CHECK-NEXT: 	const APInt r = ~x;
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt clear_sign_bit_test(const APInt &x){
// CHECK-NEXT: 	const APInt r = x;
// CHECK-NEXT: 	r.clearSignBit();
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt set_sign_bit_test(const APInt &x){
// CHECK-NEXT: 	const APInt r = x;
// CHECK-NEXT: 	r.setSignBit();
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt get_all_ones_test(const APInt &x){
// CHECK-NEXT: 	const APInt r = APInt::getAllOnes(x.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt get_signed_max_value_test(const APInt &x){
// CHECK-NEXT: 	const APInt r = APInt::getSignedMaxValue(x.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt get_signed_min_value_test(const APInt &x){
// CHECK-NEXT: 	const APInt r = APInt::getSignedMinValue(x.getBitWidth());
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
// CHECK-NEXT: const APInt reverse_bits_test(const APInt &x){
// CHECK-NEXT: 	const APInt r = x.reverseBits();
// CHECK-NEXT: 	return r;
// CHECK-NEXT: }
