import numpy as np

class MF():
    def __init__(self, ratings, hyper_params):
        self.R = np.array(ratings)
        self.num_users, self.num_items = np.shape(self.R)
        self.k = hyper_params['k']
        self.alpha = hyper_params['alpha']
        self.beta = hyper_params['beta']
        self.iterations = hyper_params['iterations']
        self.verbose = hyper_params['verbose']

        item_id_index = []
        index_item_id = []

        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)

        user_id_index = []
        index_user_id = []

        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)

    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []

        for x, y in zip(xs,ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x,y] - prediction)
            self.errors.append(self.R[x,y] - prediction)

        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - (self.beta * self.b_u[i]))
            self.b_d[j] += self.alpha * (e - (self.beta * self.b_d[j]))

            self.P[i,:] += self.alpha * ((e * self.Q[j,:]) - (self.beta * self.P[i,:]))
            self.Q[j,:] += self.alpha * ((e * self.P[i,:]) - (self.beta * self.Q[j,:]))

    def get_prediction(self):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i,:].dot(self.Q[j,:].T)
        return prediction

    def set_test(self, ratings_test):
        test_set = []
        for i in range(len(ratings_test)):
            x = self.user_id_index[ratings_test.iloc[i,0]]
            y = self.item_id_index[ratings_test.iloc[i,1]]
            z = ratings_test.iloc[i,2]
            test_set.append([x,y,z])
            self.R[x,y] = 0
        self.test_set = test_set
        return  test_set

    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))

    def test(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i,j]) for i, j in zip(rows, columns)]

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i+1, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print('Iteration : %d ; Train RMSE = %.4f ; Test RMSE = %4f' % (i + 1, rmse1, rmse2))
        return training_process

    def get_one_prediction(self, user_id, item_id):
        return self.get_prediction(self.user_id_index[user_id], self.item_id_index[item_id])

    def full_prediction(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_d[np.newaxis, :] + self.P.dot(self.Q.T)

