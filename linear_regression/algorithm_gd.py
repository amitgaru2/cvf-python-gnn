def forward(X, params):
    return [params["m"] * i + params["c"] for i in X]


def loss_fn(y, y_pred):
    N = len(y)
    return (1 / N) * sum((y[i] - y_pred[i]) ** 2 for i in range(N))


def r2_score(y, y_mean, y_pred):
    N = len(y)
    rss = sum((y[i] - y_pred[i]) ** 2 for i in range(N))
    tss = sum((y[i] - y_mean) ** 2 for i in range(N))
    r2 = 1 - rss / tss
    return r2


def gradient_m(X, y, y_pred):
    N = len(y)
    return (-2 / N) * sum((X[i] * (y[i] - y_pred[i])) for i in range(N))


def gradient_c(y, y_pred):
    N = len(y)
    return (-2 / N) * sum((y[i] - y_pred[i]) for i in range(N))


def get_iteration_vs_accuracy_data(
    X, y, iterations, X_test=None, y_test=None, L=0.0001, m=0, c=0
):
    if X_test is None:
        X_test = X
        y_test = y

    # y_mean = sum(y) / len(y)
    y_test_mean = sum(y_test) / len(y_test)

    # initial guess
    m = 0
    c = 0

    params = {"m": m, "c": c}

    steps_data = []
    accuracy_data = []

    for i in range(1, iterations + 1):
        steps_data.append(i)
        y_pred = forward(X, params)
        y_test_pred = forward(X_test, params)

        accuracy = r2_score(y_test, y_test_mean, y_test_pred)
        accuracy_data.append(accuracy)

        grad_m = gradient_m(X, y, y_pred)
        grad_c = gradient_c(y, y_pred)

        m = m - L * grad_m
        c = c - L * grad_c

        params = {"m": m, "c": c}

    return steps_data, accuracy_data
