import org.deeplearning4j.arbiter.MultiLayerSpace
import org.deeplearning4j.arbiter.conf.updater.AdamSpace
import org.deeplearning4j.arbiter.layers.DenseLayerSpace
import org.deeplearning4j.arbiter.layers.OutputLayerSpace
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace
import org.deeplearning4j.arbiter.optimize.parameter.math.MathOp
import org.deeplearning4j.arbiter.optimize.parameter.math.Op
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner
import org.deeplearning4j.arbiter.saver.local.FileModelSaver
import org.deeplearning4j.arbiter.scoring.ScoreFunctions
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.util.concurrent.TimeUnit
import java.util.logging.Logger
import kotlin.system.measureTimeMillis

/**
 * Hello-world of deep learning using deeplearning4j, Kotlin, Maven
 * Goals:
 *
 * <ol>
 * <li>Make clear which parameters are best-practice, or have the code tune the setting
 * <li>Make it run fast on GPU (assuming you have a 4GB+ GPU)
 * </ol>
 *
 * Taken from kotlin example
 * @see <a href="https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/kotlin/org/deeplearning4j/examples/feedforward/mnist">dl4j kotlin mnist</a>
 *
 * @author https://github.com/salamanders
 *
 * TODO: read latest https://github.com/deeplearning4j/Arbiter/blob/master/arbiter-deeplearning4j/src/test/java/org/deeplearning4j/arbiter/computationgraph/TestGraphLocalExecution.java
 */


private val log = Logger.getGlobal()

fun main(args: Array<String>) = measureTimeMillis {
    val saveDir = File("saves")
    if (saveDir.exists()) saveDir.delete()
    saveDir.mkdir()

    println(Nd4j.getExecutioner().executionMode())
    println(Nd4j.getExecutioner().javaClass.simpleName)

    // OPTIMIZATION, and only for GPU @see https://deeplearning4j.org/gpu
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF)

    // number of rows and columns in the input pictures.  Pixel relative position doesn't matter until using pixel adjacency
    // @see https://github.com/deeplearning4j/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/AnimalsClassification.java#L292
    val numRows = 28
    val numColumns = 28
    val outputNum = 10 // number of possible output classes (number of digits)

    // fixed hyperparameters
    val batchSize = 64 // Good in typical cases
    val rngSeed = 123456 // random number seed for reproducibility
    val numEpochs = 13 // complete training passes through the dataset.  15 initially, but lower for hyperparam search
    val activation = Activation.LEAKYRELU // Should be as-good-or-better than RELU @see https://arxiv.org/abs/1505.00853  reaches .98
    val l2Scalar = 0.005 // magic (because it doesn't matter as much)
    val outputActivation = Activation.SOFTMAX // Softmax sums to 1.  Hardmax has exactly one 1.
    // "Avoids steep learning curve by picking weights from a Gaussian distribution based on number of input neurons"
    val weightInit = WeightInit.XAVIER
    val lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD // can be bad to optimize between

    // Optimizing hyperparameters
    val learningRateHyperparam = ContinuousParameterSpace(0.0001, 0.1)  // This large a range = ok.
    // Unlikely to need much more than the avg(input, output)
    val hiddenLayerSize = DiscreteParameterSpace(outputNum, 2 * outputNum, 4 * outputNum, 8 * outputNum)

    log.info("Config complete, building model....")
    val hyperparamSpace = MultiLayerSpace.Builder()
            .seed(rngSeed.toLong())
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
            .weightInit(weightInit)
            .updater(AdamSpace(learningRateHyperparam)) // Adam usually beats raw gradient descent
            .l2(MathOp(learningRateHyperparam, Op.MUL, l2Scalar))
            .addLayer(DenseLayerSpace.Builder() // input layer
                    .nIn(numRows * numColumns)
                    .activation(activation)
                    .nOut(hiddenLayerSize)
                    .build())
            .addLayer(OutputLayerSpace.Builder() // hidden layer
                    .activation(outputActivation)
                    .lossFunction(lossFunction)
                    .nOut(outputNum)
                    .build())
            .build()

    val candidateGenerator = RandomSearchGenerator(hyperparamSpace, null)
    val dataProvider = object : DataProvider {
        override fun getDataType(): Class<*> =
                DataSetIterator::class.java

        override fun trainData(dataParameters: MutableMap<String, Any>?): Any =
                MultipleEpochsIterator(numEpochs, MnistDataSetIterator(batchSize, true, rngSeed))

        override fun testData(dataParameters: MutableMap<String, Any>?): Any =
                MnistDataSetIterator(batchSize, false, rngSeed)
    }

    val configuration = OptimizationConfiguration.Builder()
            .candidateGenerator(candidateGenerator)
            .dataProvider(dataProvider)
            .modelSaver(FileModelSaver(saveDir.absolutePath))
            .scoreFunction(ScoreFunctions.testSetAccuracy())
            .terminationConditions(listOf(
                    MaxTimeCondition(120, TimeUnit.MINUTES),
                    MaxCandidatesCondition(120)
            ))
            .build()
    println("All configurations complete.")

    val runner = LocalOptimizationRunner(configuration, MultiLayerNetworkTaskCreator())
    runner.execute()

    println("**************************************")

    println("Best score: ${runner.bestScore()}")
    println("Index of model with best score: ${runner.bestScoreCandidateIndex()}")
    println("Number of configurations evaluated: ${runner.numCandidatesCompleted()}")

    runner.results[runner.bestScoreCandidateIndex()]?.result?.let { bestResult ->
        val bestModel = bestResult.result as MultiLayerNetwork
        println("Configuration of best model:")
        println(bestModel.layerWiseConfigurations.toJson())
    }

}.let {
    println("Elapsed seconds: ${it / 1000}")
}