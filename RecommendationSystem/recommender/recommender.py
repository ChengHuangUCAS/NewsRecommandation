#coding=utf8

from algorithm.cf import cf_algorithm
from algorithm.knn import knn_algorithm


class recommender:
    def __init__(self):
        self.cf = cf_algorithm()
        self.knn = knn_algorithm()

    
    def run(self):
        print '>>>>>>run nmf<<<<<<'
        self.cf.run('nmf')
        #print '>>>>>>run userbase<<<<<<'
        #self.cf.run('user')
        print '>>>>>>run knn<<<<<<'
        self.knn.run()


if __name__ == '__main__':
    recommender = recommender()
    recommender.run()
