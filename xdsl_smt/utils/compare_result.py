class CompareResult:
    """The comparison result of a candidate transformer f and the best transformer f_best"""

    all_cases: int
    """The number of inputs"""

    sounds: int
    """The number of inputs on which f gets sound"""

    exacts: int
    """The number of inputs on which f gets exact"""

    edit_dis: int
    """The sum of edit distance between the outputs of f and the f_best """

    unsolved_cases: int
    """The number of unsolved inputs (previous transformers do not get exact)"""

    unsolved_sounds: int
    """The number of unsolved inputs on which f gets sound"""

    unsolved_exacts: int
    """The number of unsolved inputs on which f gets exact"""

    unsolved_edit_dis: int
    """The sum of edit distance between the outputs of f and the f_best on unsolved inputs"""

    cost: float | None = None

    def __init__(
        self,
        all_cases: int,
        sounds: int,
        exacts: int,
        edit_dis: int,
        unsolved_cases: int,
        unsolved_sounds: int,
        unsolved_exacts: int,
        unsolved_edit_dis: int,
        bitwidth: int,
    ):
        self.all_cases = all_cases
        self.sounds = sounds
        self.exacts = exacts
        self.edit_dis = edit_dis
        self.unsolved_cases = unsolved_cases
        self.unsolved_sounds = unsolved_sounds
        self.unsolved_exacts = unsolved_exacts
        self.unsolved_edit_dis = unsolved_edit_dis
        self.bitwidth = bitwidth

    def __str__(self):
        return f"all: {self.all_cases}\ts: {self.sounds}\te: {self.exacts}\tp: {self.edit_dis}\tunsolved:{self.unsolved_cases}\tus: {self.unsolved_sounds}\tue: {self.unsolved_exacts}\tup: {self.unsolved_edit_dis}"

    def get_cost(self) -> float:
        if self.cost is None:
            alpha = 8
            sound = self.get_sound_prop()
            dis = self.get_unsolved_edit_dis_avg() / (self.bitwidth * 2)
            self.cost = (1 - sound + alpha * dis) / (1 + alpha)
        return self.cost

    def get_sound_prop(self) -> float:
        return self.sounds / self.all_cases

    def get_exact_prop(self) -> float:
        return self.exacts / self.all_cases

    def get_unsolved_sound_prop(self) -> float:
        return self.unsolved_sounds / self.unsolved_cases

    def get_unsolved_exact_prop(self) -> float:
        return self.unsolved_exacts / self.unsolved_cases

    def get_edit_dis_avg(self) -> float:
        return self.edit_dis / self.all_cases

    def get_unsolved_edit_dis_avg(self) -> float:
        return self.unsolved_edit_dis / self.unsolved_cases

    def is_sound(self):
        return self.sounds == self.all_cases
