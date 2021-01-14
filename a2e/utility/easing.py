def ease_linear(start, end, step, total_steps):
    return start + ((end - start) / total_steps) * step


def ease_in_quad(start, end, step, total_steps):
    step_position = step / total_steps

    return start + (end - start) * step_position * step_position


def ease_out_quad(start, end, step, total_steps):
    step_position = step / total_steps

    return start + -(end - start) * step_position * (step_position - 2)
