package org.example

import java.awt.*
import java.awt.image.BufferedImage
import java.io.*
import javax.imageio.*
import javax.swing.*
import kotlin.math.exp
import kotlin.math.pow

fun logMatrix(matrix: Array<DoubleArray>, size: Int){
    for (i in 0 until  size){
        for (j in 0 until size){
            print("${matrix[i][j]} ")
        }
        print("\n")
    }
}

fun gaussFunction(
    x: Int,
    y: Int,
    sigma: Double,
    mx: Double,
    my: Double): Double = 1/(2 * Math.PI * sigma.pow(2.0)) *
        exp(-((x - mx).pow(2.0) + (y - my).pow(2.0))/ (2 * sigma.pow(2.0)))


private fun gaussKernel(size: Int, sigma: Double, mX: Double, mY: Double?): Array<DoubleArray> {
    val muY = mY?:mX
    require(!(size < 0 || size % 2 == 0)) { "Размер ядра должен быть нечетным положительным числом!" }
    val kernel = Array(size) { DoubleArray(size) }
    for (i in 0 until size) {
        for (j in 0 until size) {
            kernel[i][j] = gaussFunction(i, j, sigma, mX, muY)
        }
    }
    var sum = 0.0
    for (i in 0 until size) {
        for (j in 0 until size) {
            sum += kernel[i][j]
        }
    }
    if (sum != 0.0) {
        for (i in 0 until size) {
            for (j in 0 until size) {
                kernel[i][j] /= sum
            }
        }
    }
    logMatrix(kernel, size)
    return kernel
}

private fun convolution(image: BufferedImage, kernel: Array<DoubleArray>, size: Int): BufferedImage {
    val h = image.height
    val w = image.width
    val margin = size / 2
    val resultImage: BufferedImage = BufferedImage(w - 2 * margin, h - 2 * margin, BufferedImage.TYPE_INT_RGB)
    for (y in margin until h - margin) {
        for (x in margin until w - margin) {
            var r = 0
            var g = 0
            var b = 0
            for (y0 in y - margin until y + margin + 1) {
                for (x0 in x - margin until x + margin + 1) {
                    val pixel = Color(image.getRGB(x0, y0))
                    r += (pixel.red * kernel[y0 - (y - margin)][x0 - (x - margin)]).toInt()
                    g += (pixel.green * kernel[y0 - (y - margin)][x0 - (x - margin)]).toInt()
                    b += (pixel.blue * kernel[y0 - (y - margin)][x0 - (x - margin)]).toInt()
                }
            }
            resultImage.setRGB(x - margin, y - margin, Color(r, g, b).rgb)
        }
    }
    return resultImage
}

private fun blurImage(img: BufferedImage, size: Int, sigma: Double): BufferedImage {
    val mx = (size / 2).toDouble()
    val kernel = gaussKernel(size, sigma, mx, mx)
    val blurred: BufferedImage = convolution(img, kernel, size)
    return blurred
}

fun main() {
    val imgPath = "E:\\gaussian_blur\\src\\main\\resources\\Adler2009.jpg"
    val size = 3
    val sigma = 1.5

    try {
        val img = ImageIO.read(File(imgPath))
        val origFrame = JFrame("original")
        origFrame.add(object : Component() {
            override fun paint(g: Graphics) {
                g.drawImage(img, 0, 0, null)
            }
            override fun getPreferredSize(): Dimension {
                return Dimension(img.getWidth(null), img.getHeight(null))
            }
        })
        origFrame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        origFrame.pack()
        origFrame.setLocation(400, 100)
        origFrame.isVisible = true
        val blur: BufferedImage = blurImage(img, size, sigma)
        val blurFrame = JFrame("blurred $size x $size")
        blurFrame.add(object : Component() {
            override fun paint(g: Graphics) {
                g.drawImage(blur, 0, 0, null)
            }

            override fun getPreferredSize(): Dimension {
                return Dimension(blur.getWidth(null), blur.getHeight(null))
            }
        })
        blurFrame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        blurFrame.pack()
        blurFrame.setLocation(900, 100)
        blurFrame.isVisible = true
    } catch (e: IOException) {
        println(
            String.format("Ошибка при чтении файла: %s !", imgPath)
        )
    }
}
