using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LaboratoryWork14_15
{
    public class NeuralNetwork
    {
        private int inputSize;
        private int hiddenSize;
        private int outputSize;
        private double[,] weightsInputHidden;
        private double[,] weightsHiddenOutput;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
        }

        public void TrainAndPredict(List<double[]> trainingData, List<int> labels, int populationSize, int generations, int epochs, double learningRate)
        {
            double targetMSE = 0.001; // Заданий поріг помилки

            Console.WriteLine("Запуск генетичного алгоритму...");
            // Генетичний алгоритм: початковий підбір вагових коефіцієнтів
            var random = new Random();
            var population = new List<(double[,], double[,])>();

            // Генеруємо початкову популяцію (випадкові ваги)
            for (int i = 0; i < populationSize; i++)
            {
                double[,] wIH = new double[inputSize, hiddenSize];
                double[,] wHO = new double[hiddenSize, outputSize];

                // Заповнення ваг випадковими значеннями
                for (int x = 0; x < inputSize; x++)
                    for (int y = 0; y < hiddenSize; y++)
                        wIH[x, y] = random.NextDouble() - 0.5;

                for (int x = 0; x < hiddenSize; x++)
                    for (int y = 0; y < outputSize; y++)
                        wHO[x, y] = random.NextDouble() - 0.5;

                population.Add((wIH, wHO));
            }

            // Еволюція за допомогою генетичного алгоритму
            for (int gen = 0; gen < generations; gen++)
            {
                // Оцінюємо кожен індивідуум за функцією пристосованості
                var scores = population.Select(individual =>
                {
                    double fitness = 0.0;
                    weightsInputHidden = individual.Item1;
                    weightsHiddenOutput = individual.Item2;

                    // Обчислення функції втрат для кожного зразка
                    for (int i = 0; i < trainingData.Count; i++)
                    {
                        double[] output = Forward(trainingData[i]);
                        double target = labels[i];
                        fitness -= Math.Pow(output[0] - target, 2); // Чим менша помилка, тим краща пристосованість
                    }
                    return (fitness, individual);
                }).OrderByDescending(x => x.fitness).ToList();

                // Загальна пристосованість для рулеткової селекції
                double totalFitness = scores.Sum(x => x.fitness);

                // Рулеткова селекція для відбору кращих індивідуумів
                var selectedPopulation = new List<(double[,], double[,])>();

                for (int i = 0; i < populationSize / 2; i++)
                {
                    double randomValue = random.NextDouble() * totalFitness;
                    double cumulativeFitness = 0.0;

                    foreach (var (fitness, individual) in scores)
                    {
                        cumulativeFitness += fitness;
                        if (cumulativeFitness >= randomValue)
                        {
                            selectedPopulation.Add(individual);
                            break;
                        }
                    }
                }
                // Призначаємо відібрану популяцію як поточну популяцію
                population = selectedPopulation;

                // Мутація і створення нових вагових коефіцієнтів
                while (population.Count < populationSize)
                {
                    var parent1 = population[random.Next(population.Count)];
                    var parent2 = population[random.Next(population.Count)];

                    // Схрещення (кросовер) і мутація
                    double[,] childIH = CrossoverAndMutate(parent1.Item1, parent2.Item1, random);
                    double[,] childHO = CrossoverAndMutate(parent1.Item2, parent2.Item2, random);

                    population.Add((childIH, childHO));
                }
            }
            Console.WriteLine("Генетичний алгоритм завершено.");

            // Вибираємо кращий набір ваг після генетичного алгоритму
            var bestIndividual = population.First();
            weightsInputHidden = bestIndividual.Item1;
            weightsHiddenOutput = bestIndividual.Item2;

            Console.WriteLine("Запуск навчання за методом Backpropagation...");
            // Навчання за допомогою Backpropagation
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalMSE = 0.0; // Змінна для обчислення загальної MSE за епоху

                for (int i = 0; i < trainingData.Count; i++)
                {
                    double[] input = trainingData[i];
                    double target = labels[i];

                    // Пряме поширення
                    double[] hidden = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        hidden[h] = 0.0;
                        for (int j = 0; j < inputSize; j++)
                            hidden[h] += input[j] * weightsInputHidden[j, h];
                        hidden[h] = Sigmoid(hidden[h]);
                    }

                    double[] output = new double[outputSize];
                    for (int o = 0; o < outputSize; o++)
                    {
                        output[o] = 0.0;
                        for (int h = 0; h < hiddenSize; h++)
                            output[o] += hidden[h] * weightsHiddenOutput[h, o];
                        output[o] = Sigmoid(output[o]);
                    }

                    // Обчислення помилки і MSE для поточного зразка
                    double error = target - output[0];
                    double mse = error * error;
                    totalMSE += mse;

                    // Зворотне поширення помилки
                    double[] outputGrad = new double[outputSize];
                    for (int o = 0; o < outputSize; o++)
                    {
                        outputGrad[o] = error * SigmoidDerivative(output[o]);
                    }

                    double[] hiddenGrad = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        double sum = 0.0;
                        for (int o = 0; o < outputSize; o++)
                        {
                            sum += weightsHiddenOutput[h, o] * outputGrad[o];
                        }
                        hiddenGrad[h] = sum * SigmoidDerivative(hidden[h]);
                    }

                    // Оновлення ваг між прихованим і вихідним шарами
                    for (int h = 0; h < hiddenSize; h++)
                        for (int o = 0; o < outputSize; o++)
                            weightsHiddenOutput[h, o] += learningRate * outputGrad[o] * hidden[h];

                    // Оновлення ваг між вхідним і прихованим шарами
                    for (int inp = 0; inp < inputSize; inp++)
                        for (int h = 0; h < hiddenSize; h++)
                            weightsInputHidden[inp, h] += learningRate * hiddenGrad[h] * input[inp];
                }

                // Обчислення середнього значення MSE для поточної епохи
                totalMSE /= trainingData.Count;

                // Виведення середньоквадратичної помилки після кожної 1000-ї епохи або за умови досягнення цілі
                if (epoch % 100 == 0 || totalMSE < targetMSE)
                {
                    Console.WriteLine($"Epoch {epoch}, MSE: {totalMSE:F6}");
                }

                // Умова зупинки за MSE
                if (totalMSE < targetMSE)
                {
                    Console.WriteLine($"Зупинка навчання на епосі {epoch}, MSE досягло {totalMSE:F6}");
                    break;
                }
            }
        }

        public double[] Forward(double[] input)
        {
            // Пряме поширення
            double[] hidden = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                hidden[h] = 0.0;
                for (int i = 0; i < inputSize; i++)
                    hidden[h] += input[i] * weightsInputHidden[i, h];
                hidden[h] = Sigmoid(hidden[h]);
            }

            double[] output = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                output[o] = 0.0;
                for (int h = 0; h < hiddenSize; h++)
                    output[o] += hidden[h] * weightsHiddenOutput[h, o];
                output[o] = Sigmoid(output[o]);
            }
            return output;
        }

        private double[,] CrossoverAndMutate(double[,] parent1, double[,] parent2, Random random)
        {
            int rows = parent1.GetLength(0);
            int cols = parent1.GetLength(1);
            double[,] child = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    child[i, j] = (random.NextDouble() < 0.5) ? parent1[i, j] : parent2[i, j];
                    if (random.NextDouble() < 0.1) // Мутація
                    {
                        child[i, j] += (random.NextDouble() - 0.5) * 0.2;
                    }
                }
            }

            return child;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1.0 - x);
        }
    }
}
