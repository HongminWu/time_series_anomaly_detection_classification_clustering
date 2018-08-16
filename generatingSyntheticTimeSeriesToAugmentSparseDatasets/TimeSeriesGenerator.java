/*******************************************************************************
 * Copyright (C) 2017 Francois Petitjean
 * Contributors:
 * 	Francois Petitjean
 *      Germain Forestier
 * 
 * This file is part of 
 * "Generating synthetic time series to augment sparse datasets."
 * accepted for publication at the IEEE ICDM 2017 conference
 * 
 * "Generating synthetic time series to augment sparse datasets"
 * is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * "Generating synthetic time series to augment sparse datasets"
 * is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with "Generating synthetic time series to augment sparse datasets".
 * If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

import static java.lang.Math.sqrt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.random.RandomDataGenerator;

/**
 * This class contains static methods to create a set of synthetic time series
 * from a set of given time series
 * 
 * @author Francois Petitjean, Germain Forestier
 */
public class TimeSeriesGenerator {

	protected enum MatrixDirection {
		NOTHING, DIAGONAL, LEFT, TOP;
	}

	public static String[] methodName = new String[] { null, "AverageAll", "AverageSelected", "AveragesSelectedWithDistance" ,"AALTD'16"};

	/**
	 * This function returns a set of fake data generated from a set of time
	 * series to get inspiration from. The distance measure to induce the
	 * space is DTW.
	 * 
	 * @param methodNumber
	 *                choice of the method (1. averaging from all series
	 *                with one only a few being peaks; 2. averaging using a
	 *                few neighbors ; 3. averaging using an
	 *                exponential decay around a randomly chosen center of
	 *                gravity; 5. using AALTD'16 paper)
	 * @param originalData
	 *                the data to learn from
	 * @param nFakeDataToGenerate
	 *                the requested number of data
	 * @param seed
	 *                the seed of the random number generator (makes the
	 *                process deterministic)
	 * @return
	 */
	public static double[][] generateFakeData(int methodNumber, double[][] originalData, int nFakeDataToGenerate, long seed) {
		return generateFakeData(methodNumber, originalData, nFakeDataToGenerate, originalData[0].length, seed);
	}

	/**
	 * This function returns a set of fake data generated from a set of time
	 * series to get inspiration from. The distance measure to induce the
	 * space is DTW with a warping window.
	 * 
	 * @param methodNumber
	 *                choice of the method (1. averaging from all series
	 *                with one only a few being peaks; 2. averaging using a
	 *                few neighbors - Anonymous; 3. averaging using an
	 *                exponential decay around a randomly chosen center of
	 *                gravity; 4. generate using NLAAF and close neighbors; 5. using AALTD'16 paper)
	 * @param originalData
	 *                the data to learn from
	 * @param nFakeDataToGenerate
	 *                the requested number of data
	 * @param w
	 *                warping window
	 * @param seed
	 *                the seed of the random number generator (makes the
	 *                process deterministic)
	 * @return
	 */
	public static double[][] generateFakeData(int methodNumber, double[][] originalData, int nFakeDataToGenerate, int w, long seed) {
		switch (methodNumber) {
		case 1:
			return generateFakeDataAverageAll(originalData, nFakeDataToGenerate, 0.2, w, seed);
		case 2:
			return generateFakeAverageSelected(originalData, nFakeDataToGenerate, w, seed);
		case 3:
			return generateFakeDataAverageSelectedWithDistance(originalData, nFakeDataToGenerate, w, seed);
		case 4:
			return generateFakeDataAALTD(originalData, nFakeDataToGenerate, seed);
		default:
			throw new RuntimeException("Method should be between 1 and 4");
		}

	}

	/**
	 * This function returns a set of fake data generated from a set of time
	 * series to get inspiration from.
	 * 
	 * @param originalData
	 *                the data to learn from
	 * @param nFakeDataToGenerate
	 *                the requested number of data
	 * @param seed
	 *                the seed of the random number generator (makes the
	 *                process deterministic)
	 * @return
	 */
	private static double[][] generateFakeDataAverageAll(double[][] originalData, int nFakeDataToGenerate, int w, long seed) {
		return generateFakeDataAverageAll(originalData, nFakeDataToGenerate, 1.0, w, seed);
	}

	/**
	 * This function returns a set of fake data generated from a set of time
	 * series to get inspiration from.
	 * 
	 * @param originalData
	 *                the data to learn from
	 * @param nFakeDataToGenerate
	 *                the requested number of data
	 * @param seed
	 *                the seed of the random number generator (makes the
	 *                process deterministic)
	 * @return
	 */
	private static double[][] generateFakeDataAverageAll(double[][] originalData, int nFakeDataToGenerate, double spread, int w, long seed) {
		if (originalData == null || originalData.length == 0) {
			throw new RuntimeException("no data to generate from");
		} else if (nFakeDataToGenerate == 0) {
			return new double[0][];
		}
		int nOriginalData = originalData.length;
		int length = originalData[0].length;
		double[][] res = new double[nFakeDataToGenerate][length];
		double[][] warpingMatrix = new double[length][length];
		MatrixDirection[][] choiceMatrix = new MatrixDirection[length][length];
		double[] weights = new double[nOriginalData];
		RandomDataGenerator r = new RandomDataGenerator();

		for (int i = 0; i < nFakeDataToGenerate; i++) {

			// sample random weights for each time series of the
			// original data
			int indexMax = -1;
			double maxWeight = 0;
			double sumWeights = 0;
			for (int s = 0; s < weights.length; s++) {
				weights[s] = r.nextGamma(spread, 1.0);
				if (weights[s] > maxWeight) {
					maxWeight = weights[s];
					indexMax = s;
				}
				sumWeights += weights[s];
			}
			for (int s = 0; s < weights.length; s++) {
				weights[s] /= sumWeights;
			}
			// System.out.println(Arrays.toString(weights));

			// choose one sequence from the original data
			double[] sample = originalData[indexMax];

			res[i] = DBAMeanWeighted(sample, originalData, weights, w, warpingMatrix, choiceMatrix);

		}

		return res;
	}

	/**
	 * This function returns a set of fake data generated from a set of time
	 * series to get inspiration from. 1. select key point 2. select a pool
	 * of m out of the k nearest neighbours (m=2, k=5 feels right) 3.
	 * generate a weighted average - perhaps 50% key, 30% nearer of pool,
	 * 20% other.
	 * 
	 * @param originalData
	 *                the data to learn from
	 * @param nFakeDataToGenerate
	 *                the requested number of data
	 * @param seed
	 *                the seed of the random number generator (makes the
	 *                process deterministic)
	 * @return
	 */
	private static double[][] generateFakeAverageSelected(double[][] originalData, int nFakeDataToGenerate, int w, long seed) {
		if (originalData == null || originalData.length == 0) {
			throw new RuntimeException("no data to generate from");
		} else if (nFakeDataToGenerate == 0) {
			return new double[0][];
		}
		int nOriginalData = originalData.length;
		int length = originalData[0].length;
		double[][] res = new double[nFakeDataToGenerate][length];
		double[][] warpingMatrix = new double[length][length];
		MatrixDirection[][] choiceMatrix = new MatrixDirection[length][length];
		RandomDataGenerator r = new RandomDataGenerator();
		final int k = Math.min(5, originalData.length - 1);
		final int subk = Math.min(k, 2);
		final double weightKey = .5;
		final double weightNeighbors = .3;
		final short nIterDBA = 5;
		int nOthers = originalData.length - 1 - subk;
		Neighbor<double[]>[] topK = new Neighbor[k];
		double[] weights = new double[nOriginalData];

		// compute matrix of distances
		float[][] distances = new float[nOriginalData][];
		for (int i = 1; i < nOriginalData; i++) {
			distances[i] = new float[i];
			for (int j = 0; j < i; j++) {
				distances[i][j] = (float) DTW(originalData[i], originalData[j], w, warpingMatrix);
			}
		}

		for (int i = 0; i < nFakeDataToGenerate; i++) {

			// 1. sample one point
			int keyPointIndex = r.nextInt(0, originalData.length - 1);
			double[] keyPoint = originalData[keyPointIndex];

			// put default weight of remaining weight over the whole
			// dataset (as
			// if)
			double remainingWeight = 1.0 - weightKey - weightNeighbors;
			if (nOthers == 0) {
				Arrays.fill(weights, 0.0);
			} else {
				Arrays.fill(weights, remainingWeight / (originalData.length - 1 - subk));
			}

			weights[keyPointIndex] = weightKey;

			// 2.a find k nearest neighbors
			Arrays.fill(topK, new Neighbor<double[]>(null, -1, Double.POSITIVE_INFINITY));
			for (int j = 0; j < originalData.length; j++) {
				if (j == keyPointIndex)
					continue;
				double[] s = originalData[j];

				int iTmp = keyPointIndex, jTmp = j;
				if (iTmp < jTmp) {
					int z = iTmp;
					iTmp = jTmp;
					jTmp = z;
				}

				double d = distances[iTmp][jTmp];
				if (d < topK[topK.length - 1].distToQuery) {
					int insertionPoint = topK.length - 1;
					while (insertionPoint > 0 && d < topK[insertionPoint].distToQuery) {
						topK[insertionPoint] = topK[insertionPoint - 1];
						insertionPoint--;
					}
					Neighbor<double[]> newNeighbor = new Neighbor<double[]>(s, j, d);
					topK[insertionPoint] = newNeighbor;
				}
			}

			// 2.b select finalk neighbors out of topk
			int[] selectIndexes = r.nextPermutation(k, subk);
			for (int j = 0; j < selectIndexes.length; j++) {
				int indexInTrain = topK[selectIndexes[j]].indexInTrain;
				weights[indexInTrain] = weightNeighbors / subk;
			}

			// System.out.println(Arrays.toString(weights));

			res[i] = keyPoint;
			for (int iter = 0; iter < nIterDBA; iter++) {
				res[i] = DBAMeanWeighted(res[i], originalData, weights, w, warpingMatrix, choiceMatrix);
			}

		}

		return res;
	}

	/**
	 * This function returns a set of fake data generated from a set of time
	 * series to get inspiration from. This follows the paper "Data
	 * Augmentation for Time Series Classification using Convolutional
	 * Neural Networks" presented at AALTD'16 (ECML/PKDD workshop). The
	 * method implemented is WW.
	 * 
	 * @param originalData
	 *                the data to learn from
	 * @param nFakeDataToGenerate
	 *                the requested number of data
	 * @param seed
	 *                the seed of the random number generator (makes the
	 *                process deterministic)
	 * @return
	 */
	private static double[][] generateFakeDataAALTD(double[][] originalData, int nFakeDataToGenerate, long seed) {
		if (originalData == null || originalData.length == 0) {
			throw new RuntimeException("no data to generate from");
		} else if (nFakeDataToGenerate == 0) {
			return new double[0][];
		}
		int nOriginalData = originalData.length;
		int length = originalData[0].length;
		double[][] res = new double[nFakeDataToGenerate][];
		Random r = new Random(seed);
		int sliceLength = (int) (0.1 * length);

		for (int i = 0; i < nFakeDataToGenerate; i++) {
			// 1. sample one time series
			int indexToModify = r.nextInt(nOriginalData);
			double[] toModify = originalData[indexToModify];

			int indexStartSlice = r.nextInt(length - sliceLength);
			ArrayList<Double> fake = new ArrayList<>();
			for (int l = 0; l < indexStartSlice; l++) {
				fake.add(toModify[l]);
			}
			boolean speedup = r.nextBoolean();
			if (speedup) {// speed up by 2
				for (int l = indexStartSlice; l < indexStartSlice + sliceLength; l += 2) {
					fake.add(toModify[l]);
				}
			} else {// slow down by 2
				for (int l = indexStartSlice; l < indexStartSlice + sliceLength - 1; l++) {
					fake.add(toModify[l]);
					fake.add((toModify[l] + toModify[l + 1]) / 2.0);
				}
			}
			for (int l = indexStartSlice + sliceLength; l < length; l++) {
				fake.add(toModify[l]);
			}
			res[i] = uniformScaling(fake, length);
		}
		return res;

	}

	private static double[] uniformScaling(ArrayList<Double> fake, int length) {
		int originalLength = fake.size();
		double[] res = new double[length];
		for (int l = 0; l < res.length; l++) {
			double positionInOriginal = 1.0 * originalLength * l / length;
			int indexPreviousPoint = (int) positionInOriginal;
			int indexNextPoint = indexPreviousPoint + 1;
			if (indexNextPoint >= fake.size()) {// shouldn't happen
				res[l] = fake.get(indexPreviousPoint);
			} else {
				double weightPreviousPoint = 1.0 - (positionInOriginal - indexPreviousPoint);
				double value = fake.get(indexPreviousPoint) * weightPreviousPoint
								+ fake.get(indexNextPoint) * (1.0 - weightPreviousPoint);
				res[l] = value;
			}
		}
		return res;
	}

	
	/**
	 * This function returns a set of fake data generated from a set of time
	 * series to get inspiration from. It differs from
	 * {@link #generateFakeData(double[][], int, long)} in the way that we
	 * now try to have weights that are calculated with a center of gravity
	 * taken as one of the time series. This should generate more variance
	 * 
	 * @param originalData
	 *                the data to learn from
	 * @param nFakeDataToGenerate
	 *                the requested number of data
	 * @param seed
	 *                the seed of the random number generator (makes the
	 *                process deterministic)
	 * @return
	 */
	private static double[][] generateFakeDataAverageSelectedWithDistance(double[][] originalData, int nFakeDataToGenerate, int w, long seed) {
		if (originalData == null || originalData.length == 0) {
			throw new RuntimeException("no data to generate from");
		} else if (nFakeDataToGenerate == 0) {
			return new double[0][];
		}
		int nOriginalData = originalData.length;
		int length = originalData[0].length;
		double[][] res = new double[nFakeDataToGenerate][length];
		double[][] warpingMatrix = new double[length][length];
		MatrixDirection[][] choiceMatrix = new MatrixDirection[length][length];
		double[] weights = new double[nOriginalData];
		Random r = new Random(seed);
		final short nIterDBA = 5;

		// compute matrix of distances
		float[][] distances = new float[nOriginalData][];
		for (int i = 1; i < nOriginalData; i++) {
			distances[i] = new float[i];
			for (int j = 0; j < i; j++) {
				distances[i][j] = (float) DTW(originalData[i], originalData[j], w, warpingMatrix);
			}
		}

		for (int i = 0; i < nFakeDataToGenerate; i++) {

			/*
			 * choose one sequence to be the center of gravity.
			 * Choice should be made avoiding points of low density
			 */
			int indexCenterOfGravity = r.nextInt(originalData.length);

			double[] centerOfgravity = originalData[indexCenterOfGravity];

			// find NN of the center of gravity in original data
			float bestDist = Float.POSITIVE_INFINITY;
			for (int s = 0; s < nOriginalData; s++) {
				if (s != indexCenterOfGravity) {
					int iTmp = s, jTmp = indexCenterOfGravity;
					if (iTmp < jTmp) {
						int k = iTmp;
						iTmp = jTmp;
						jTmp = k;
					}
					if (distances[iTmp][jTmp] < bestDist) {
						bestDist = distances[iTmp][jTmp];
					}
				}
			}
			// System.out.println("distance to NN="+bestDist);

			/*
			 * now we're going to set the exponential decay such
			 * that the nearest neighbor gets a weight of 50%
			 */

			for (int s = 0; s < weights.length; s++) {
				if (indexCenterOfGravity == s) {
					weights[s] = 1.0;
				} else {
					int iTmp = s, jTmp = indexCenterOfGravity;
					if (iTmp < jTmp) {
						int k = iTmp;
						iTmp = jTmp;
						jTmp = k;
					}
					float dist = distances[iTmp][jTmp];
					float ratioDistToNN = dist / bestDist;
					weights[s] = Math.exp(Math.log(0.5) * ratioDistToNN);
					// System.out.println("dist="+dist+"\tratio="+ratioDistToNN+"\tweight="+weights[s]);
				}
			}

			res[i] = centerOfgravity;
			for (int iter = 0; iter < nIterDBA; iter++) {
				res[i] = DBAMeanWeighted(res[i], originalData, weights, w, warpingMatrix, choiceMatrix);
			}

		}

		return res;
	}

	/**
	 * Full DTW
	 * 
	 * @param seq1
	 * @param seq2
	 * @param warpingMatrix
	 * @return
	 */
	private static final double DTW(double[] seq1, double[] seq2, double[][] warpingMatrix) {
		return DTW(seq1, seq2, seq1.length, warpingMatrix);
	}

	/**
	 * DTW with warping window
	 * 
	 * @param seq1
	 * @param seq2
	 * @param w
	 * @param warpingMatrix
	 * @return
	 */
	private static final double DTW(double[] seq1, double[] seq2, int w, double[][] warpingMatrix) {

		final int length1 = seq1.length;
		final int length2 = seq2.length;

		int i, j;
		warpingMatrix[0][0] = squaredDistance(seq1[0], seq2[0]);
		for (i = 1; i < Math.min(length1, 1 + w); i++) {
			warpingMatrix[i][0] = warpingMatrix[i - 1][0] + squaredDistance(seq1[i], seq2[0]);
		}

		for (j = 1; j < Math.min(length2, 1 + w); j++) {
			warpingMatrix[0][j] = warpingMatrix[0][j - 1] + squaredDistance(seq1[0], seq2[j]);
		}
		if (j < length2) {
			warpingMatrix[0][j] = Double.POSITIVE_INFINITY;
		}

		for (i = 1; i < length1; i++) {
			int jStart = Math.max(1, i - w);
			int jStop = Math.min(length2, i + w + 1);
			warpingMatrix[i][jStart - 1] = Double.POSITIVE_INFINITY;

			for (j = jStart; j < jStop; j++) {
				warpingMatrix[i][j] = Tools.Min3(warpingMatrix[i - 1][j - 1], warpingMatrix[i][j - 1], warpingMatrix[i - 1][j])
								+ squaredDistance(seq1[i], seq2[j]);
			}
			if (jStop < length2) {
				warpingMatrix[i][jStop] = Double.POSITIVE_INFINITY;
			}
		}

		return sqrt(warpingMatrix[length1 - 1][length2 - 1]);
	}

	private static final double DTW(ArrayList<Double> seq1, double[] seq2, int w, double[][] warpingMatrix) {

		final int length1 = seq1.size();
		final int length2 = seq2.length;

		int i, j;
		warpingMatrix[0][0] = squaredDistance(seq1.get(0), seq2[0]);
		for (i = 1; i < Math.min(length1, 1 + w); i++) {
			warpingMatrix[i][0] = warpingMatrix[i - 1][0] + squaredDistance(seq1.get(i), seq2[0]);
		}

		for (j = 1; j < Math.min(length2, 1 + w); j++) {
			warpingMatrix[0][j] = warpingMatrix[0][j - 1] + squaredDistance(seq1.get(0), seq2[j]);
		}
		if (j < length2) {
			warpingMatrix[0][j] = Double.POSITIVE_INFINITY;
		}

		for (i = 1; i < length1; i++) {
			int jStart = Math.max(1, i - w);
			int jStop = Math.min(length2, i + w + 1);
			warpingMatrix[i][jStart - 1] = Double.POSITIVE_INFINITY;

			for (j = jStart; j < jStop; j++) {
				warpingMatrix[i][j] = Tools.Min3(warpingMatrix[i - 1][j - 1], warpingMatrix[i][j - 1], warpingMatrix[i - 1][j])
								+ squaredDistance(seq1.get(i), seq2[j]);
			}
			if (jStop < length2) {
				warpingMatrix[i][jStop] = Double.POSITIVE_INFINITY;
			}
		}

		return sqrt(warpingMatrix[length1 - 1][length2 - 1]);
	}

	private static final double DTW(ArrayList<Double> seq1, double[] seq2, double[][] warpingMatrix) {
		return DTW(seq1, seq2, seq1.size(), warpingMatrix);
	}

	private synchronized static final double[] DBAMean(final double[] oldCenter, final double[][] tabSequence, int w, double[][] warpingMatrix,
					MatrixDirection[][] choiceMatrix) {
		double[] weights = new double[tabSequence.length];
		Arrays.fill(weights, 1.0);
		return DBAMeanWeighted(oldCenter, tabSequence, weights, w, warpingMatrix, choiceMatrix);
	}

	private synchronized static final double[] DBAMean(final double[] oldCenter, final double[][] tabSequence, double[][] warpingMatrix,
					MatrixDirection[][] choiceMatrix) {
		double[] weights = new double[tabSequence.length];
		Arrays.fill(weights, 1.0);
		return DBAMeanWeighted(oldCenter, tabSequence, weights, oldCenter.length, warpingMatrix, choiceMatrix);
	}

	private synchronized static final double[] DBAMeanWeighted(final double[] oldCenter, final double[][] tabSequence, final double[] weights,
					double[][] warpingMatrix, MatrixDirection[][] choiceMatrix) {
		return DBAMeanWeighted(oldCenter, tabSequence, weights, oldCenter.length, warpingMatrix, choiceMatrix);
	}

	private synchronized static final double[] DBAMeanWeighted(final double[] oldCenter, final double[][] tabSequence, final double[] weights,
					int w, double[][] warpingMatrix, MatrixDirection[][] choiceMatrix) {

		double[] sumWeights = new double[oldCenter.length];
		int i, j, indiceRes;
		double res = 0.0;
		final int length1 = oldCenter.length;
		double[] resMean = new double[oldCenter.length];

		for (int s = 0; s < tabSequence.length; s++) {
			if (weights[s] == 0.0)
				continue;
			double[] S = tabSequence[s];
			int length2 = S.length;

			warpingMatrix[0][0] = squaredDistance(oldCenter[0], S[0]);
			choiceMatrix[0][0] = MatrixDirection.NOTHING;

			for (i = 1; i < Math.min(length1, 1 + w); i++) {
				warpingMatrix[i][0] = warpingMatrix[i - 1][0] + squaredDistance(oldCenter[i], S[0]);
				choiceMatrix[i][0] = MatrixDirection.TOP;
			}
			for (j = 1; j < Math.min(length2, 1 + w); j++) {
				warpingMatrix[0][j] = warpingMatrix[0][j - 1] + squaredDistance(S[j], oldCenter[0]);
				choiceMatrix[0][j] = MatrixDirection.LEFT;
			}
			if (j < length2) {
				warpingMatrix[0][j] = Double.POSITIVE_INFINITY;
			}

			for (i = 1; i < length1; i++) {
				int jStart = Math.max(1, i - w);
				int jStop = Math.min(length2, i + w + 1);
				warpingMatrix[i][jStart - 1] = Double.POSITIVE_INFINITY;

				for (j = jStart; j < jStop; j++) {
					indiceRes = ArgMin3(warpingMatrix[i - 1][j - 1], warpingMatrix[i][j - 1], warpingMatrix[i - 1][j]);
					choiceMatrix[i][j] = intToMatrixDirection(indiceRes);
					switch (choiceMatrix[i][j]) {
					case DIAGONAL:
						res = warpingMatrix[i - 1][j - 1];
						break;
					case LEFT:
						res = warpingMatrix[i][j - 1];
						break;
					case TOP:
						res = warpingMatrix[i - 1][j];
						break;
					case NOTHING:
						throw new RuntimeException("Found a non-coded path in the matrix;should never happen...");
					}
					warpingMatrix[i][j] = res + squaredDistance(oldCenter[i], S[j]);
				}
				if (jStop < length2) {
					warpingMatrix[i][jStop] = Double.POSITIVE_INFINITY;
				}
			}

			i = length1 - 1;
			j = length2 - 1;

			while (i > 0 || j > 0) {
				resMean[i] += S[j] * weights[s];
				sumWeights[i] += weights[s];
				switch (choiceMatrix[i][j]) {
				case DIAGONAL:
					i = i - 1;
					j = j - 1;
					break;
				case LEFT:
					j = j - 1;
					break;
				case TOP:
					i = i - 1;
					break;
				case NOTHING:
					throw new RuntimeException("Found a non-coded path in the matrix;should never happen...");
				}

			}
			resMean[0] += S[0] * weights[s];
			sumWeights[0] += weights[s];

		}

		for (int t = 0; t < resMean.length; t++) {
			resMean[t] /= sumWeights[t];
		}

		return resMean;
	}

	private synchronized static final double[] NLAAFMeanWeightedRescaled(final double[] seq1, double weight1, final double[] seq2, double weight2,
					int w, double[][] warpingMatrix, MatrixDirection[][] choiceMatrix) {
		double sumWeights = weight1 + weight2;
		int i, j, indiceRes;
		double res = 0.0;
		ArrayList<Double> resMean = new ArrayList<>();
		int length1 = seq1.length;
		int length2 = seq2.length;

		warpingMatrix[0][0] = squaredDistance(seq1[0], seq2[0]);
		choiceMatrix[0][0] = MatrixDirection.NOTHING;

		for (i = 1; i < Math.min(length1, 1 + w); i++) {
			warpingMatrix[i][0] = warpingMatrix[i - 1][0] + squaredDistance(seq1[i], seq2[0]);
			choiceMatrix[i][0] = MatrixDirection.TOP;
		}
		for (j = 1; j < Math.min(length2, 1 + w); j++) {
			warpingMatrix[0][j] = warpingMatrix[0][j - 1] + squaredDistance(seq2[j], seq1[0]);
			choiceMatrix[0][j] = MatrixDirection.LEFT;
		}
		if (j < length2) {
			warpingMatrix[0][j] = Double.POSITIVE_INFINITY;
		}

		for (i = 1; i < seq1.length; i++) {
			int jStart = Math.max(1, i - w);
			int jStop = Math.min(length2, i + w + 1);
			warpingMatrix[i][jStart - 1] = Double.POSITIVE_INFINITY;

			for (j = jStart; j < jStop; j++) {
				indiceRes = ArgMin3(warpingMatrix[i - 1][j - 1], warpingMatrix[i][j - 1], warpingMatrix[i - 1][j]);
				choiceMatrix[i][j] = intToMatrixDirection(indiceRes);
				switch (choiceMatrix[i][j]) {
				case DIAGONAL:
					res = warpingMatrix[i - 1][j - 1];
					break;
				case LEFT:
					res = warpingMatrix[i][j - 1];
					break;
				case TOP:
					res = warpingMatrix[i - 1][j];
					break;
				case NOTHING:
					throw new RuntimeException("Found a non-coded path in the matrix;should never happen...");
				}
				warpingMatrix[i][j] = res + squaredDistance(seq1[i], seq2[j]);
			}
			if (jStop < length2) {
				warpingMatrix[i][jStop] = Double.POSITIVE_INFINITY;
			}
		}

		i = seq1.length - 1;
		j = seq2.length - 1;

		while (i > 0 || j > 0) {
			double barycenter = (seq1[i] * weight1 + seq2[j] * weight2) / sumWeights;
			resMean.add(0, barycenter);
			switch (choiceMatrix[i][j]) {
			case DIAGONAL:
				i = i - 1;
				j = j - 1;
				break;
			case LEFT:
				j = j - 1;
				break;
			case TOP:
				i = i - 1;
				break;
			case NOTHING:
				throw new RuntimeException("Found a non-coded path in the matrix;should never happen...");
			}
		}
		double barycenter = (seq1[0] * weight1 + seq2[0] * weight2) / sumWeights;
		resMean.add(0, barycenter);

		/*
		 * now doing adaptive scaling. try to remove one element at a
		 * time until reached the size of the data
		 */
		while (resMean.size() > seq1.length) {
			double bestScore = Double.POSITIVE_INFINITY;
			int bestIndexToRemove = -1;
			ArrayList<Double> copyMean = (ArrayList<Double>) resMean.clone();
			for (int t = 0; t < resMean.size(); t++) {
				Double removed = copyMean.remove(t);
				double tmpDist = DTW(resMean, seq1, w, warpingMatrix);
				double score = tmpDist * tmpDist;
				tmpDist = DTW(resMean, seq2, warpingMatrix);
				score += tmpDist * tmpDist;
				if (score < bestScore) {
					bestIndexToRemove = t;
					bestScore = score;
				}
				copyMean.add(t, removed);
			}
			resMean.remove(bestIndexToRemove);
		}

		// now transform the mean to a tab
		double[] resMeanTab = new double[resMean.size()];
		for (int k = 0; k < resMeanTab.length; k++) {
			resMeanTab[k] = resMean.get(k);
		}

		return resMeanTab;
	}

	private static final double squaredDistance(double a, double b) {
		double diff = (a - b);
		return diff * diff;
	}

	private final static int ArgMin3(final double a, final double b, final double c) {
		return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
	}

	private final static MatrixDirection intToMatrixDirection(int n) {
		switch (n) {
		case 0:
			return MatrixDirection.DIAGONAL;
		case 1:
			return MatrixDirection.LEFT;
		case 2:
			return MatrixDirection.TOP;
		default:
			return MatrixDirection.NOTHING;
		}
	}

}

class Neighbor<K> implements Comparable<Neighbor<K>>{
	public K neighbor;
	public int indexInTrain;
	public double distToQuery;
	
	public Neighbor(K neighbor,int indexInTrain,double distToQuery){
		this.neighbor = neighbor;
		this.indexInTrain = indexInTrain;
		this.distToQuery = distToQuery;
	}
	
	@Override
	public int compareTo(Neighbor<K> o) {
		return Double.compare(distToQuery, o.distToQuery);
	}
}
