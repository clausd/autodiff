require 'test/unit'
require './lib/autodiff'
require './lib/autodiff/mallet'
require 'pry'

class MalletTest < Test::Unit::TestCase

  def test_solver
    x = ::Autodiff::Variable.new()
    y = ::Autodiff::Variable.new()

    f = x*x*(-3) - y*y*4 + x*2 - y*4 + 18

    solver = ::Autodiff::Mallet::Simple.new(f)
    x.set(0)
    y.set(0)
    puts f.value
    puts solver.solve([0.0,0.0])
    puts f.value
    puts x.value
    puts y.value

  end
end
