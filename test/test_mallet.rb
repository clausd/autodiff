require 'test/unit'
require './lib/autodiff'
require './lib/autodiff/mallet'
require 'pry'

class MalletTest < Test::Unit::TestCase

  def test_interface

    x = ::Autodiff::Variable.new()
    y = ::Autodiff::Variable.new()

    f = x*x*(-3) - y*y*4 + x*2 - y*4 + 18

    solver = ::Autodiff::Mallet::Simple.new(f)
    x.set(3.0)
    y.set(2.0)

    buffer = []
    solver.getParameters(buffer)
    assert_equal [3.0, 2.0], buffer
    assert_equal 2, solver.getNumParameters
    assert_equal 2.0, solver.getParameter(1)
    assert_equal (-3*3*3-4*2*2+2*3-4*2+18).to_f, solver.getValue
    solver.getValueGradient(buffer)
    assert_equal [-16.0, -20.0], buffer
    solver.setParameters([5.0,6.0])
    assert_equal [5.0, 6.0], [x.value, y.value]
    solver.setParameter(0,23.0)
    assert_equal 23.0, x.value
  end

  def test_solver
    x = ::Autodiff::Variable.new()
    y = ::Autodiff::Variable.new()

    f = x*x*(-3) - y*y*4 + x*2 - y*4 + 18

    solver = ::Autodiff::Mallet::Simple.new(f)
    x.set(0)
    y.set(0)
    puts f.value
    puts solver.solve
    puts f.value
    puts x.value
    puts y.value

  end
end
