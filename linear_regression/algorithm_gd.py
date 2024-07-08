def forward(x, m, c):
    return [m * i + c for i in x]


def r2_score(y_actual, y_actual_mean, y_pred):
    N = len(y_actual)
    rss = sum((y_actual[i] - y_pred[i]) ** 2 for i in range(N))
    tss = sum((y_actual[i] - y_actual_mean) ** 2 for i in range(N))
    r2 = 1 - rss / tss
    return r2


def gradient_m(x, y_actual, y_pred):
    N = len(y_actual)
    return (-2 / N) * sum((x[i] * (y_actual[i] - y_pred[i])) for i in range(N))


def gradient_c(y_actual, y_pred):
    N = len(y_actual)
    return (-2 / N) * sum((y_actual[i] - y_pred[i]) for i in range(N))


def get_iteration_vs_accuracy_data(x, y, iterations, L=0.0001, m=0, c=0):
    y_mean = sum(y) / len(y)

    # initial guess
    m = 0
    c = 0

    steps_data = []
    accuracy_data = []

    for i in range(1, iterations + 1):
        steps_data.append(i)
        y_pred = forward(x, m, c)
        accuracy = r2_score(y, y_mean, y_pred)
        accuracy_data.append(accuracy)
        # print("Loss:", loss, "| Accuracy:", accuracy)

        grad_m = gradient_m(x, y, y_pred)
        grad_c = gradient_c(y, y_pred)

        m = m - L * grad_m
        c = c - L * grad_c

    return steps_data, accuracy_data
