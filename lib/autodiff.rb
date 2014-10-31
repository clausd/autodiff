require "autodiff/version"

class Matrix
  unless Matrix.respond_to?(:row_count)
    def row_count
      row_size
    end

    def column_count
      column_size
    end
  end
end

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

    def flush

    end

    def arguments
      raise "Not implemented"
    end

    ## overloads for arithmetic

    def +(y)
      Plus.new(self, Term.construct(y))
    end

    # TODO Minus-operator for less overhead
    def -(y)
      Plus.new(self, Times.new(Term.construct(y),Term.construct(-1)))
    end

    def *(y)
      Times.new(self,Term.construct(y))
    end

    # TODO Divided-operator for less overhead
    def /(y)
      Times.new(self,Power.new(Term.construct(v),Term.construct(-1)))
    end

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

    def arguments
      []
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

    def arguments
      [self]
    end

  end

  class Matrix < Term

    def set(m)
      unless m.nil?
        @gradient = ::Matrix.zero(m.row_count, m.column_count)
        @m = m
      end
    end

    def value
      @m
    end

    def transposed_value
      @m.transpose
    end

    def gradient
      @gradient
    end

    def accumulate(xbar)
      if xbar.is_a?(::Matrix) # this is a bit speculative - need to write down a proof
        @gradient = @gradient + xbar
      end
    end

    def arguments
      [self]
    end

  end

  class Vector < Matrix
    # just simplified construction, otherwise implemented in matrix...

    # TODO
  end

  # collapse a dimension by summing
  class Reduce < Term

    def initialize(m,dimension)
      unless m.nil?
        @m = m
        @reduce_by = dimension
      end
    end

    def reduce_rows
      ::Matrix.build(1, @m.value.column_count) { |i,j| @m.value.column(j).inject(&:+)}
    end

    def reduce_columns
      ::Matrix.build(@m.value.row_count, 1) { |i,j| @m.value.row(i).inject(&:+)}
    end

    def value
      case @reduce_by
        when :rows
          reduce_rows
        when :columns
          reduce_columns
      end
    end

    def transposed_value
      value.transpose
    end

    def gradient

    end

    def accumulate(xbar) # watch the value
      case @reduce_by
        when :rows
          @m.accumulate(::Matrix.build(@m.value.row_count, @m.value.column_count) { |i,j| xbar[0,j]})
        when :columns
          @m.accumulate(::Matrix.build(@m.value.row_count, @m.value.column_count) { |i,j| xbar[i,0]})
      end
    end

    def arguments
      [@m]
    end

  end

  class Transpose < Term

    def initialize(m)
      unless m.nil?
        @m = m
      end
    end

    def value
      @m.transposed_value
    end

    def transposed_value
      @m.value
    end

    def gradient

    end

    def accumulate(xbar)
      @m.accumulate(xbar.transpose)
    end

    def arguments
      [@m]
    end

  end

  # compute a term for each element in  a matrix
  # possibly better to have a set list of scalar functions that we generate classes for
  # I think that's WAY more viable...
  class Apply < Term

    def initialize(matrix, term)
      @m = matrix
      @term = term
    end

    # TODO - store a recompute-signal
    def value
      @m.value.collect {|v| @term.arguments.first.set(v); @term.value}
    end

    def transposed_value
      value.transpose
    end

    # accumulation is linear, so we can simple trust the accumulation logic from @term
    def accumulate(xbar)
      @m.accumulate(::Matrix.build(@m.value.row_count, @m.value.column_count) {|i,j| @term.arguments.first.set(@m.value[i,j]); @term.accumulate(xbar[i,j]); @term.arguments.first.gradient})
    end

    def arguments
      [@m]
    end

  end

  # turn n by m things into k by l things when n*m == k*l
  class Reshape < Term

  end

  class Plus < Term

    def initialize(x,y)
      @x = x #x.attach(self)
      @y = y #y.attach(self)
    end

    def value(*args)
      @x.value + @y.value
    end

    # TODO - too costly
    def transposed_value
      if value.respond_to?(:transpose)
        value.transpose
      else
        value
      end
    end

    def accumulate(xbar)
      @x.accumulate(xbar)
      @y.accumulate(xbar)
    end

    def arguments
      (@x.arguments + @y.arguments).uniq
    end

  end

  class Times < Term

    def initialize(x,y)
      @x = x #x.attach(self)
      @y = y #y.attach(self)

    end

    def value(*args)
      @x.value * @y.value
    end

    # TODO - too costly
    def transposed_value
      if value.respond_to?(:transpose)
        value.transpose
      else
        value
      end
    end

    def gradient
    end

    # this works for scalars as well as matrices
    def accumulate(xbar)
      @x.accumulate(@y.transposed_value*xbar)
      @y.accumulate(xbar*@x.transposed_value)
    end

    def arguments
      (@x.arguments + @y.arguments).uniq
    end

  end

  class Power < Term

    def initialize(x,y)
      @x = x #x.attach(self)
      @y = y #y.attach(self)
    end

    def value(*args)
      @x.value**@y.value
    end

    def transposed_value(*args)
      @x.value**@y.value
    end

    def gradient
    end

    # this does not support matrix values
    def accumulate(xbar)
      @x.accumulate(@y.value*@x.value**(@y.value-1)*xbar)
      @y.accumulate(Math.log(@x.value)*self.value*xbar)
    end

    def arguments
      (@x.arguments + @y.arguments).uniq
    end

  end

end
