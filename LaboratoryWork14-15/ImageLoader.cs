using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Media;

namespace LaboratoryWork14_15
{
    public class ImageLoader
    {
        public static (List<double[]>, List<int>) LoadDogCatData(string rootFolder)
        {
            var data = new List<double[]>();
            var labels = new List<int>();
            var classNames = new Dictionary<string, int> { { "dog", 1 }, { "cat", 0 } };

            foreach (var folder in Directory.GetDirectories(rootFolder))
            {
                var className = Path.GetFileName(folder);
                if (!classNames.ContainsKey(className))
                {
                    Console.WriteLine($"Пропускаємо папку: {className}");
                    continue;
                }

                var label = classNames[className];
                foreach (var imagePath in Directory.GetFiles(folder, "*.jpg"))
                {
                    Console.WriteLine($"Завантаження зображення: {imagePath}");
                    var pixels = LoadImageAsVector(imagePath);
                    data.Add(pixels);
                    labels.Add(label);
                }
            }
            return (data, labels);
        }

        public static double[] LoadImageAsVector(string imagePath)
        {
            using (Bitmap bitmap = new Bitmap(imagePath))
            {
                using (Bitmap resizedBitmap = new Bitmap(bitmap, new Size(32, 32)))
                {
                    double[] pixels = new double[32 * 32 * 3];

                    for (int y = 0; y < resizedBitmap.Height; y++)
                    {
                        for (int x = 0; x < resizedBitmap.Width; x++)
                        {
                            System.Drawing.Color pixel = resizedBitmap.GetPixel(x, y);
                            int index = (y * resizedBitmap.Width + x) * 3;
                            pixels[index] = pixel.R / 255.0;
                            pixels[index + 1] = pixel.G / 255.0;
                            pixels[index + 2] = pixel.B / 255.0;
                        }
                    }
                    return pixels;
                }
            }
        }
    }
}
