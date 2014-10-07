require "autodiff/version"

module Autodiff

  class Term

    @parent = nil

    def self.construct(x)
      k = x.class
      case
      when k <= Term
        x
      when k == Fixnum
        Constant.new(x)
      when Float
        Constant.new(x)
      when Array
        if x.first.is_a?(Array)
          Matrix.new(x)
        else
          Vector.new(x)
        end
      else
        raise "Can't wrap " + x.class + " as Term"
      end
    end

    def attach(parent)
      @parent = parent
      self
    end

    def value(*args)
      raise 'Not implemented'
    end

    def gradient
      raise 'Not implemented'
    end

    ## overloads for arithmetic

    def +(y)
      Plus.new(self, Term.construct(y))
    end

    # def -(y)
    #   Minus.new(self, v)
    # end

    def *(y)
      Times.new(self,Term.construct(y))
    end

    # def /(y)
    #   DividedBy.new(self,v)
    # end
    #
    def **(y)
      Power.new(self,Term.construct(y))
    end

  end

  class Constant < Term

    def initialize(x)
      @x = x unless x.nil?
    end

    def value
      @x
    end

    def transposed_value
      @x
    end

    def gradient
      0
    end

    def accumulate(xbar)
    end

  end

  class Variable < Term

    def set(x)
      unless x.nil?
        @gradient = 0
        @x = x
      end
    end

    def value
      @x
    end

    def transposed_value
      @x
    end

    def gradient
      @gradient
    end

    def accumulate(xbar)
      @gradient += xbar
    end

  end

  class Matrix < Term

    def set(m)
      unless m.nil?
        @gradient = Matrix.zero(m.row_count, m.column_count)
        @m = m
      end
    end

    def value
      @m
    end

    def transposed_value
      @m.transpose
    end

    def accumulate(xbar)
      if xbar.is_a?(::Matrix) # this is a bit speculative - need to write down a proof
        @gradient = @gradient + xbar
      end
    end
  end

  class Trace < Term

    def initialize(m)
      unless m.nil?
        @m = m
      end
    end

    def value
      @m[0,0]
    end

    def transposed_value
      @m[0,0]
    end

    def gradient

    end

    def accumulate(xbar)
      @m.accumulate(::Matrix.build(@m.row_count, @m.column_count) { |i,j| xbar})
    end

  end

  class Transpose < Term

    def initialize(m)
      unless m.nil?
        @m = m.transpose
      end
    end

    def value
      @m
    end

    def transposed_value
      @m.transpose
    end

    def gradient

    end

    def accumulate(xbar)
      @m.accumulate(xbar.transpose)
    end

  end

  class Vector < Matrix
    # just simplified construction, otherwise implemented in matrix...

    # TODO
  end

  class Plus < Term

    def initialize(x,y)
      @x = x.attach(self)
      @y = y.attach(self)
    end

    def value(*args)
      @x.value + @y.value
    end

    def gradient
    end

    def accumulate(xbar)
      @x.accumulate(xbar)
      @y.accumulate(xbar)
    end

  end

  class Times < Term

    def initialize(x,y)
      @x = x.attach(self)
      @y = y.attach(self)

    end

    def value(*args)
      @x.value * @y.value
    end

    def gradient
    end

    # this works for scalars as well as matrices
    def accumulate(xbar)
      @x.accumulate(@y.transposed_value*xbar)
      @y.accumulate(xbar*@x.transposed_value)
    end

  end

  class Power < Term

    def initialize(x,y)
      @x = x.attach(self)
      @y = y.attach(self)
    end

    def value(*args)
      @x.value**@y.value
    end

    def gradient
    end

    # this does not support matrix values
    def accumulate(xbar)
      @x.accumulate(@y.value*@x.value**(@y.value-1)*xbar)
      @y.accumulate(Math.log(@x.value)*self.value*xbar)
    end

  end

end
