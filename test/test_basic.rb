require 'test/unit'
require './lib/autodiff'
require 'pry'

class BasicTest < Test::Unit::TestCase
  def test_eval
    x = ::Autodiff::Variable.new()
    y = ::Autodiff::Variable.new()
    expression = x**2+x*y
    x.set(5)
    y.set(4)
    p (x**2).value
    assert 45 == expression.value
    # binding.pry
    # expression.value(4,5)
    # expression.gradient(4,5)
  end
end
