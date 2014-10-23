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

  def test_norm
    
  end

end
