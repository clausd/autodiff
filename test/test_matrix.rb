require 'test/unit'
require './lib/Autodiff'
require 'pry'
require 'matrix'

class MatrixTest < Test::Unit::TestCase

  def test_product
    m = ::Autodiff::Matrix.new
    n = ::Autodiff::Matrix.new
    product = m*n
    m.set(Matrix[[1,0],[0,2]])
    n.set(Matrix[[1,2],[3,4]])
    assert_equal product.value, Matrix[[1,2],[6,8]]
  end

  def test_gradient
    v1 = ::Autodiff::Matrix.new #column vector
    dotproduct = ::Autodiff::Times.new(::Autodiff::Transpose.new(v1),v1)
    v1.set(::Matrix[[1,2,3,4,5]])
    #p dotproduct.value.to_a
    dotproduct.accumulate(Matrix[[1]])
    assert_equal v1.gradient, 2*::Matrix[[1,2,3,4,5]]
  end

  def test_apply
    power = ::Autodiff::Variable.new**2
    m = ::Autodiff::Matrix.new
    mpower = ::Autodiff::Apply.new(m,power)
    m.set(Matrix[[1,2],[3,4]])
    assert_equal mpower.value, Matrix[[1,4],[9,16]]
    mpower.accumulate(Matrix[[1,1],[1,1]])
    assert_equal m.gradient, Matrix[[2,4],[6,8]]
    # binding.pry
  end

  def test_reduce
    m = ::Autodiff::Matrix.new
    onedim = ::Autodiff::Reduce.new(m, :rows)
    m.set(Matrix[[1,2],[3,4]])
    assert_equal Matrix[[4,6]], onedim.value
    zerodim = ::Autodiff::Reduce.new(onedim, :columns)
    assert_equal Matrix[[10]], zerodim.value
    # TODO gradient
  end

  def test_norm
    m = ::Autodiff::Matrix.new
    power = ::Autodiff::Variable.new**2.0
    norm_squared = ::Autodiff::Reduce.new(::Autodiff::Reduce.new(::Autodiff::Apply.new(m,power), :rows),:columns)
    m.set(Matrix[[1,1],[1,1]])
    # p norm_squared.value
    norm_squared.accumulate(Matrix[[1]])
    # p m.gradient
  end

end
