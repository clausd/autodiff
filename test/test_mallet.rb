require 'test/unit'
require './lib/autodiff'
require './lib/autodiff/mallet'
require 'pry'

class MalletTest < Test::Unit::TestCase

  def test_interface

    x = AD.x
    y = AD.x

    f = AD.k(-3)*x*x + AD.k(-4)*y*y + x*2 - y*4 + 18
    f.arrange
    solver = Autodiff::Mallet::Simple.new(f)

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
    x = AD.x
    y = AD.x

    f = AD.k(-3)*x*x + AD.k(-4)*y*y + x*2 - y*4 + 18
    f.arrange
    solver = Autodiff::Mallet::Simple.new(f)
    x.set(0)
    y.set(0)

    assert solver.solve
    # puts f.value
    # puts x.value
    # puts y.value

  end

  def test_matrix_solver
    dim = 10
    m = AD.M(dim,dim)
    c = AD.K((1..(dim*dim)).map {|i| i}.each_slice(dim).to_a)
    random_matrix = AD.K((1..(dim*dim)).map {|i| rand+i/(dim*dim)}.each_slice(dim).to_a)
    power = AD.x**2.0
    sigmoid = AD.k(1)/(AD.k(Math::E)**(AD.k(-1)*AD.x) + 1)

    norm_squared = Autodiff::Pick.new(
      Autodiff::Reduce.new(
        Autodiff::Reduce.new(
          Autodiff::Apply.new(Autodiff::Apply.new(m*c,sigmoid)-random_matrix,power),
          :rows),
        :columns),0,0)*(-1)


    norm_squared.arrange
    solver = Autodiff::Mallet::Simple.new(norm_squared)
    m.set((1..dim*dim).map {0.0})
    norm_squared.accumulate(1)

    # p norm_squared.partials

    t = Time.now
    # e = norm_squared.evals
    begin
      solver.solve
    rescue Exception => e
      p e
    end
    puts "TIME " + (Time.now.to_f-t.to_f).to_s
    # puts norm_squared.evals-e
    puts m.value
    puts norm_squared.value
  end

end
