// RUN: xdsl-smt "%s" | xdsl-smt | filecheck "%s"

builtin.module {
    %state = "smt.declare_const"() : () -> !effect.state
    // CHECK:           %{{.*}} = "smt.declare_const"() : () -> !effect.state
    %y = ub_effect.trigger %state
    // CHECK-NEXT:      %{{.*}} = ub_effect.trigger %{{.*}}
    %b = ub_effect.to_bool %y
    // CHECK-NEXT:      %{{.*}} = ub_effect.to_bool %{{.*}}
}
