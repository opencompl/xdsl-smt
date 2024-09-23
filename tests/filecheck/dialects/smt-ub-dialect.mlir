// RUN: xdsl-smt "%s" | xdsl-smt | filecheck "%s"

builtin.module {
    %x = smt_ub.create_state
    // CHECK:           %{{.*}} = smt_ub.create_state
    %y = smt_ub.trigger %x
    // CHECK-NEXT:      %{{.*}} = smt_ub.trigger %{{.*}}
    %b = smt_ub.to_bool %y
    // CHECK-NEXT:      %{{.*}} = smt_ub.to_bool %{{.*}}
}
