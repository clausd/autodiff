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
end
