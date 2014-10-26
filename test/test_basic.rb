require 'test/unit'
require './lib/autodiff'
require 'pry'

class BasicTest < Test::Unit::TestCase
  def test_eval
    x = ::Autodiff::Variable.new()
    y = ::Autodiff::Variable.new()
    expression = x**2+x*y
    ddx = x*2+y
    ddy = x

    x.set(5)
    y.set(4)
    assert 45 == expression.value
    expression.accumulate(1)
    assert x.gradient == ddx.value
    assert y.gradient == ddy.value

    x.set(0)
    y.set(0)
    assert 0 == expression.value
    expression.accumulate(1)
    assert x.gradient == ddx.value
    assert y.gradient == ddy.value


    x.set(1)
    y.set(1)
    assert 2 == expression.value
    expression.accumulate(1)
    assert x.gradient == ddx.value
    assert y.gradient == ddy.value

  end

  def test_sigmoid_function
    x = ::Autodiff::Variable.new()
    sigmoid = (::Autodiff::Constant.new(Math::E)**(x*-1.0) + 1)**-1.0
    sigmoid_derivative = sigmoid*(sigmoid-1)*-1
    x.set(0.0)
    assert_in_delta(0.5,sigmoid.value, 0.00001)
    sigmoid.accumulate(1)
    assert_in_delta(sigmoid_derivative.value,x.gradient, 0.00001)
    x.set(-Math.log(3))
    assert_in_delta(0.25,sigmoid.value, 0.00001)
    sigmoid.accumulate(1)
    assert_in_delta(sigmoid_derivative.value,x.gradient, 0.00001)
    x.set(Math.log(3))
    assert_in_delta(0.75,sigmoid.value, 0.00001)
    sigmoid.accumulate(1)
    assert_in_delta(sigmoid_derivative.value,x.gradient, 0.00001)
  end

end
