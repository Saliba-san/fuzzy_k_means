import numpy as np

def euclid_dist(data, centroids):
    """
    Calcula distancia euclidiana de uma centroid aos pontos

    Argumentos:
        data -> List[Tuple[int, int]]
        centroids -> List[Tuple[int, int]]

    Retorno:
        np.ndarray
    """
    centroids_array = np.array(centroids)
    data_array = np.array(data)
    difference = np.subtract(np.expand_dims(data_array, axis=1), np.expand_dims(centroids_array, axis=0))
        
    return np.sqrt(np.sum(np.square(difference), axis=2))

def grau_pertencimento(distancias, m):
    """
    Normaliza cada linha da matriz para obter o grau de pertencimento de cada ponto a cada cluster

    Argumentos:
        matrix -> np.ndarray 

    Retorno:
        np.ndarray
    """
    membership = 1.0 / distancias ** (2.0/(m-1))
    membership /= np.sum(membership, axis=1)[:, np.newaxis]
    return membership

def fuzzy_kmeans(X, k, max_iter, m):
    """
    Calcula os clusters, com centroides e pontos pertencentes a cada um

    Argumentos:
        X -> List[Tuple[int, int]]: lista de pontos
        k -> int: number of klusters
        max_iter -> int: maximum iterations
    """
    # Initialize centroids 
    centroids = np.random.uniform(0, 1, (k, X.shape[1]))

    # Run the main k-means algorithm
    for i in range(max_iter):
        distancias = euclid_dist(X, centroids)
        pertencimento = grau_pertencimento(distancias, m)

        old_centroids = centroids.copy()
        centroids = np.dot(pertencimento.T, X)
        centroids /= pertencimento.sum(axis=0)[:, np.newaxis]
        
        print(pertencimento)
