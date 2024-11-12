using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace LaboratoryWork14_15
{
    public partial class MainWindow : Window
    {
        private NeuralNetwork neuralNetwork;
        private string loadedImagePath;

        public MainWindow()
        {
            InitializeComponent();

            // Ініціалізація нейронної мережі з 32*32*3 вхідними нейронами (для RGB-зображення),
            // 64 прихованими нейронами і 1 вихід ("кіт" або "собака").
            neuralNetwork = new NeuralNetwork(32 * 32 * 3, 64, 1);
        }

        private async void TrainNetwork(object sender, RoutedEventArgs e)
        {
            // Вимикаємо кнопки до завершення навчання
            LoadImageButton.IsEnabled = false;
            PredictButton.IsEnabled = false;

            // Завантаження навчальних даних (дані для котів і собак).
            var (trainingData, labels) = ImageLoader.LoadDogCatData("D:\\2024-2025\\МСШІ\\Лабораторна робота 14-15\\LaboratoryWork14-15\\LaboratoryWork14-15\\TrainingData");

            Console.WriteLine($"Кількість завантажених зображень: {trainingData.Count}");
            Console.WriteLine($"Кількість завантажених міток: {labels.Count}");

            // Параметри для генетичного алгоритму та Backpropagation.
            int populationSize = 20;
            int generations = 50;
            int epochs = 1000;
            double learningRate = 0.1;

            // Запуск навчання мережі з переданими параметрами.
            await Task.Run(() => neuralNetwork.TrainAndPredict(trainingData, labels, populationSize, generations, epochs, learningRate));

            // Після завершення навчання включаємо кнопки
            LoadImageButton.IsEnabled = true;
            PredictButton.IsEnabled = true;
        }

        private void LoadTestImage_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog();
            openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
            if (openFileDialog.ShowDialog() == true)
            {
                loadedImagePath = openFileDialog.FileName;
                TestImage.Source = new BitmapImage(new Uri(loadedImagePath));
            }
        }

        private void Predict_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(loadedImagePath))
            {
                MessageBox.Show("Будь ласка, завантажте зображення для тестування.");
                return;
            }

            // Очищуємо ResultText перед новим прогнозом.
            ResultText.Text = string.Empty;

            // Завантажуємо вектор для нового зображення.
            double[] inputVector = ImageLoader.LoadImageAsVector(loadedImagePath);

            // Отримуємо прогноз від нейронної мережі.
            var prediction = neuralNetwork.Forward(inputVector);

            Console.WriteLine(prediction[0]);

            ResultText.Text = prediction[0] > 0.5 ? "Це собака!" : "Це кіт!";
        }
    }
}
