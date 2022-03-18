def grad_descent_v2(f, df, low=None, high=None, callback=None):
    """ 
    Реализация градиентного спуска для функций с несколькими локальным минимумами,
    но с известной окрестностью глобального минимума. 
    Все тесты будут иметь такую природу.
    :param func: float -> float — функция 
    :param deriv: float -> float — её производная
    :param low: float — левая граница окрестности
    :param high: float — правая граница окрестности
    :param callback: callalbe -- функция логирования
    """
    def find_local_min(f, df, low_local, high_local, iters=5000, lr=0.05):
        x0 = np.random.uniform(low_local, high_local)
        x = x0

        for i in range(1, iters + 1):
            x = x - lr * df(x) * 1/(iters**0.33)
            x = np.clip(x, low_local, high_local)
            callback(x, f(x))
        return x

    dist = (high - low) / 6
    first = low
    second = low + dist
    a = np.zeros(6)

    for i in range(6):
        #for j in range(10):
        a[i] = find_local_min(f, df, first, second, iters=5000, lr=0.05)
    first += dist
    second += dist


    # вам нужно запустить find_local_min несколько раз с разными границами и среди полученных ответов выбрать тот, при котором f имеет наименьшее значение 
    # подсказка: np.argmin
    # YOUR CODE
    
    # Разбейте отрезок [low,high] на 3-6 равных частей 

    # Для каждой части запустите find_local_min несколько 
    # (преподавательский код запускает 10) раз
        
    best_estimate = a[np.argmin(a)]
    
    return best_estimate
