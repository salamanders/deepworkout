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
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace
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
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.util.concurrent.TimeUnit
import java.util.logging.Logger
import kotlin.system.measureTimeMillis

/**
 * Hello-world of deep learning using deeplearning4j, Kotlin, Gradle
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
 * TODO: read latest https://github.com/deeplearning4j/Arbiter/blob/ae0c37e470e4650ae2eb312afbf939773037ee0b/arbiter-deeplearning4j/src/test/java/org/deeplearning4j/arbiter/computationgraph/TestGraphLocalExecution.java#L223
 */


private val log = Logger.getGlobal()

fun main(args: Array<String>) = measureTimeMillis {
    //println(Nd4j.getExecutioner().executionMode())
    //println(Nd4j.getExecutioner().javaClass.simpleName)

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

    // Optimizing hyperparameters
    val learningRateHyperparam = ContinuousParameterSpace(0.0001, 0.1) // Are these sane bounds?
    // Should be based on another param, @see https://github.com/deeplearning4j/Arbiter/issues/123
    val l2Hyperparam = MathOp(learningRateHyperparam, Op.MUL, 0.005)
    val layerSizeHyperparam = IntegerParameterSpace(outputNum, numRows * numColumns) // Is this necessary to optimize?
    // Q: what is the difference?
    val lossFunction = DiscreteParameterSpace(
            LossFunctions.LossFunction.MCXENT,
            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD
    )
    val saveDir = File("saves")
    if (saveDir.exists()) saveDir.delete()
    saveDir.mkdir()

    log.info("Config complete, building model....")
    val hyperparamSpace = MultiLayerSpace.Builder()
            .seed(rngSeed.toLong())
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE) // TODO enable when released
            .weightInit(WeightInit.XAVIER) // "Avoids steep learning curve by picking weights from a Gaussian distribution based on number of input neurons"
            .updater(AdamSpace(learningRateHyperparam))
            .l2(l2Hyperparam)
            .addLayer(DenseLayerSpace.Builder() // input layer
                    .nIn(numRows * numColumns)
                    .activation(activation)
                    .nOut(layerSizeHyperparam)
                    .build())
            .addLayer(OutputLayerSpace.Builder() // hidden layer
                    .activation(Activation.SOFTMAX) // Softmax sums to 1.  Hardmax has exactly one 1.
                    .lossFunction(lossFunction)
                    .nOut(outputNum)
                    .build())
            .build()


    val configuration = OptimizationConfiguration.Builder()
            .candidateGenerator(RandomSearchGenerator(hyperparamSpace, null))
            .dataProvider(object : DataProvider {
                override fun getDataType(): Class<*> =
                        DataSetIterator::class.java

                override fun trainData(dataParameters: MutableMap<String, Any>?): Any =
                        MultipleEpochsIterator(numEpochs, MnistDataSetIterator(batchSize, true, rngSeed))

                override fun testData(dataParameters: MutableMap<String, Any>?): Any =
                        MnistDataSetIterator(batchSize, false, rngSeed)
            })
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