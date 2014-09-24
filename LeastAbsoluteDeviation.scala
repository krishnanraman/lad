import org.apache.commons.math3.distribution.TDistribution
import org.apache.commons.math3.optim.linear._
import org.apache.commons.math3.optim._
import scala.collection.JavaConverters._

/**
  SEE: http://en.wikipedia.org/wiki/Least_absolute_deviations
  Fits a straight line given bunch of (x,y) coordinates via Least Absolute Deviation ( NOT OLS )
  LAD solutions are obtained using a Simplex solver.
  If you have n points, the Simplex solver constructs a system of 2n+1 equations in (n+2) unknowns!

  eg. To fit a straight line through 100 points, need to solve 201 equations with 102 unknowns!

  ie. compures m & c for the best LAD fit y = mx+c
  Computes predictions based on y = mx+c
  Computes 95% confidence interval for prediction using the T distribution
*/
object LeastAbsoluteDeviation extends App {
  type D = Double

  println( "Usage Example: ")
  println("Random Dataset around 2x + 3")

  val xy = (0.0 to 100.0 by 0.2).map{ x=> (x,2 * x + 3 + math.random/10.0) }
  println(xy)
  val lad = LeastAbsoluteDeviation(xy)
  println("Model: " + lad.fit.format(3).mkStr)
  println( lad.predict(4.0))
}

import LeastAbsoluteDeviation.D
case class ConfidenceInterval(lowerBound:D, upperBound:D)
case class Prediction(value:D, ci:ConfidenceInterval)

trait regress {
  def fit: Line
  def predict(x:D): Prediction
}

case class LeastAbsoluteDeviation(xy:Seq[(D,D)]) extends regress {
  assert(xy.size > 2)

  def n = xy.size

  val line:Option[Line] = Some(mkfit)
  def fit:Line = line.get

  def mkfit:Line = {

    // min(u1 + u2 + .... + un) = min(u1 + u2 + .... + un + 0a + 0b)
    // make an array with n+2 members
    // n of these are 1, the last 2 are 0
    val objarr = Array.fill[D](n)(1.0) ++ Array(0.0,0.0)
    val objfunc = new LinearObjectiveFunction( objarr, 0.0)

    val constraints = xy.zipWithIndex.map {
        ptIdx =>
        val (pt,idx) = ptIdx
        val (x,y) = pt
        val weights = Array.tabulate[D](n)(i=> if(i==idx) 1.0 else 0.0)
        val a1 = weights ++ Array(x, 1.0)
        val a2 = weights ++ Array(-x, -1.0)
        val c1 = new LinearConstraint(a1, Relationship.GEQ, y)
        val c2 = new LinearConstraint(a2, Relationship.GEQ, -y)
        List(c1,c2)
      }.flatten

    // debug: constraints.foreach(c=> println(c.getCoefficients.toArray.mkString(",")))
    val constraintSet = new LinearConstraintSet( constraints.asJava )
    val res = new SimplexSolver().optimize(objfunc, constraintSet).getFirst.drop(n)
    Line(res(0), res(1))
  }

  def yMean = xy.map { case (x,y) => y}.sum/(n+0.0)

  def xMean = xy.map { case (x,y) => x}.sum/(n+0.0)

  // aka standard error, root mean square error, rmse, ...
  def se = standardErrorResiduals
  def rmse = standardErrorResiduals
  def rootMeanSquareError = standardErrorResiduals

  def standardErrorResiduals = {
    val mse = xy.map { case (x,y) => (y, line.get.apply(x))}
      .map{ case(yi,yhat) => (yi - yhat)*(yi-yhat) }
      .sum / (n - 2)
    math.sqrt(mse)
  }

  def sey = standardErrorPrediction _
  def standardErrorPrediction(xk:D) = {
      se * math.sqrt((1+1.0/n + ((xk - xMean)*(xk-xMean)/xerror)))
  }

  def xerror = {
    xy.map { case (x,y) => (x, xMean)}
      .map{ case(xi,xbar) => (xi - xbar)*(xi-xbar) }
      .sum
  }

  def standardErrorSlope = {
    se/math.sqrt(xerror)
  }

  def degreesOfFreedom = n-2

  def tDist = new TDistribution(degreesOfFreedom)

  def tstat = tDist.inverseCumulativeProbability(0.975) // 95% Confidence Interval, 2-sided test

  def predict(x:D): Prediction = {
    val yvalue = line.get.apply(x)
    val err = tstat * sey(x)
    val lower = yvalue - err
    val upper = yvalue + err
    Prediction(yvalue, ConfidenceInterval(lower, upper))
  }
}

case class Line(slope:D, intercept:D) {

  def format(numberOfDecimals:Int):Line = {
    val formatter = "%."+numberOfDecimals+"f"
    val res = Seq(slope,intercept).map{ x=> formatter.format(x) }.map{ x=> x.toDouble }
    Line(res(0), res(1))
  }

  def mkStr = List(slope, intercept)
  .zip(List("x",""))
  .map { i=> i._1 + i._2}
  .mkString(" + ")
  .replace("+ -", "-")

  def apply = {x:D => slope*x + intercept}
}
