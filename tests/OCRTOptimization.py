import unittest
from functools import partial

import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_sampling, get_mutation, get_termination
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pyswarms.single import GlobalBestPSO

from ORCT1 import compute_orct1
import cv2

from ORCT1Inverse import compute_orct1inverse
from ORCT2 import compute_orct2
from ORCT2Inverse import compute_orct2inverse
from ORCT2Plus3 import compute_orct2plus3
from ORCT2Plus3Inverse import compute_orct2plus3inverse
from Utils.Evaluation import Evaluation
from Utils.DataUtils import DataUtils


class TestORCTOptimization(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.datasetUtils = DataUtils()
        self.evaluation = Evaluation()

    def opt_func(self, X, twoComplement):
        n_particles = X.shape[0]  # number of particles
        costs = []
        for x in X:
            x = np.reshape(x, [2, 2])
            filtered = compute_orct2(compute_orct1(twoComplement, x), x)
            filtered = (filtered + 255) / 2

            def inverseFunction(data):
                data = data.astype('float32') * 2 - 255
                data = compute_orct2inverse(data, x)
                data = compute_orct1inverse(data, x)
                return data

            sampleFunctionReverse = inverseFunction
            psnr, ssim, jpeg2000CompressionRatioAfter, jpeg2000CompressionRatioBefore = self.evaluation.evaluate(filtered, twoComplement, sampleFunctionReverse)
            cost = np.abs((1 / psnr) * (1 / ssim) * 1 / (jpeg2000CompressionRatioAfter))
            costs.append(cost)
        return np.array(costs)

    def test_orct12_optimization(self):
        bayer = self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        options = {'c1': 0.5, 'c2': 0.1, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=10, dimensions=4, options=options)

        costFunction = partial(self.opt_func, twoComplement=twoComplement)
        cost, pos = optimizer.optimize(costFunction, iters=30)

        pass

    def test_orct12_(self):
        pos = np.asarray([0.39182592, 0.23747258, 0.51497874, -0.08751142])
        x = np.reshape(pos, [2, 2])
        # x = np.asarray([[0.05011018, -0.53709484],
        #                 [-1.1104253, -0.30699651]])
        bayer = self.datasetUtils.readCFAImages()

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2(compute_orct1(twoComplement, x), x)

        filtered = (filtered + 255) / 2

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 255
            data = compute_orct2inverse(data, x)
            data = compute_orct1inverse(data, x)
            return data

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, twoComplement, sampleFunctionReverse)
        pass

    def test_ocrtOptimizedWithDataset(self):
        pos = np.asarray([0.39182592, 0.23747258, 0.51497874, -0.08751142])
        x = np.reshape(pos, [2, 2])
        rgbImages = self.datasetUtils.loadKodakDataset()
        cfaImages, image_size = self.datasetUtils.convertDatasetToCFA(rgbImages)
        bayer = cfaImages[2, :, :]

        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")

        filtered = compute_orct2plus3(compute_orct1(twoComplement, x), x)

        filtered = (filtered + 255) / 2

        def inverseFunction(data):
            data = data.astype('float32') * 2 - 255
            data = compute_orct2plus3inverse(data, x)
            data = compute_orct1inverse(data, x)
            return data

        sampleFunctionReverse = inverseFunction
        self.evaluation.evaluate(filtered, bayer, sampleFunctionReverse)
        pass

    def test_multiObjOptimization(self):
        algorithm = NSGA2(
            pop_size=10,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
        termination = get_termination("n_gen", 10)
        bayer = self.datasetUtils.readCFAImages()
        twoComplement = self.datasetUtils.twoComplementMatrix(bayer)
        twoComplement = twoComplement.astype("float32")
        problem = MyProblem(twoComplement)
        res = minimize(problem,
                       algorithm,
                       termination,
                       save_history=True,
                       verbose=True)

        # Objective Space
        res.F = 1 / res.F
        plot = Scatter(title="Objective Space")
        plot.add(res.F)
        plot.show()
        print("Best filter{}".format(np.reshape(res.opt[-1].X, [2, 2])))


class MyProblem(Problem):

    def __init__(self, twoComplement):
        super().__init__(n_var=4,
                         n_obj=3,
                         n_constr=2,
                         xl=np.array([-2, -2, -2, -2]),
                         xu=np.array([2, 2, 2, 2]))
        self.twoComplement = twoComplement
        self.evaluation = Evaluation()

    def _evaluate(self, X, out, *args, **kwargs):
        n_particles = X.shape[0]  # number of particles
        psnrs = []
        ssims = []
        jpeg200compressions = []
        for x in X:
            x = np.reshape(x, [2, 2])
            filtered = compute_orct2(compute_orct1(self.twoComplement, x), x)
            filtered = (filtered + 255) / 2

            def inverseFunction(data):
                data = data.astype('float32') * 2 - 255
                data = compute_orct2inverse(data, x)
                data = compute_orct1inverse(data, x)
                return data

            sampleFunctionReverse = inverseFunction
            try:
                psnr, ssim, jpeg2000CompressionRatioAfter, jpeg2000CompressionRatioBefore = self.evaluation.evaluate(filtered, self.twoComplement, sampleFunctionReverse, verbose=False)
            except:
                psnr = 1
                ssim = .1
                jpeg2000CompressionRatioAfter = 1
                jpeg2000CompressionRatioBefore = 1.001
            psnrs.append(1 / psnr)
            ssims.append(1 / ssim)
            CR = jpeg2000CompressionRatioAfter - jpeg2000CompressionRatioBefore
            if CR < 0:
                CR = 0.001
            jpeg200compressions.append(1 / CR)

        out["F"] = np.column_stack([psnrs, ssims, jpeg200compressions])
        out["G"] = np.column_stack([-1 * np.asarray(psnrs), -1 * np.asarray(ssims)])
