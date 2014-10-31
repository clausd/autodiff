require "autodiff/version"

# this is silly just live with the stupidity of size
# going to make a matrix-adapter though, so I can switch out classes dep on platform (MRI vs JRuby)
class Matrix
  unless Matrix.respond_to?(:row_count)
    def row_count
      row_size
    end

    def column_count
      column_size
    end
  end

  def []=(i, j, x)
    @rows[i][j] = x
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

    def value_array
      arguments.map(&:value_array).flatten
    end

    def gradient_array
      arguments.map(&:value_array).flatten
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
        @x = [x].flatten.first
      end
    end

    def setParams(i, array)
      set(array[i])
    end

    def setParams(array)
      set(array[0])
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

    def value_array
      [self.value]
    end

    def gradient_array
      [self.gradient]
    end
  end

  class Matrix < Term

    def set(m)
      unless m.nil?
        if m.respond_to?(:to_a)
          mm = m.to_a
        else
          mm = m
        end
        if mm.is_a?(Array)
          if mm.first.is_a?(Array)
            @m = ::Matrix.rows(mm)
          else
            @m = ::Matrix.rows(mm.each_slice(@m.row_count).to_a)
          end
        elsif mmm.is_a?(::Matrix)
          @m = mm
        else
          raise "not implemented " + m.class
        end
        @gradient = ::Matrix.zero(@m.row_count, @m.column_count)
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

    def value_array
      self.value.to_a.flatten
    end

    def gradient_array
      self.gradient.to_a.flatten
    end

  end

  class ConstantMatrix < Matrix

    def initialize(m)
      unless m.nil?
        if m.is_a?(Array)
          if m.first.is_a?(Array)
            @m = ::Matrix.rows(m)
          else
            @m = ::Matrix.rows(m.each_slice(@m.row_count))
          end
        elsif m.is_a?(::Matrix)
          @m = m
        else
          raise "not implemented"
        end
      end
    end

    def accumulate(xbar)
    end

    def arguments
      []
    end

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
      @m.arguments
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
      @m.arguments
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
      @m.arguments
    end

  end

  # get a scalar from a matrix
  class Pick < Term
    def initialize(matrix, i, j)
      @m = matrix
      @i = i
      @j = j
    end

    # TODO - store a recompute-signal
    def value
      @m.value[@i,@j]
    end

    def transposed_value
      value
    end

    # accumulation is linear, so we can simple trust the accumulation logic from @term
    def accumulate(xbar)
      @m.accumulate(::Matrix.build(@m.value.row_count, @m.value.column_count) {|i,j| (i == @i && j == @j ? xbar : 0.0) })
    end

    def arguments
      @m.arguments
    end
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
      @y.accumulate(Math.log(@x.value)*self.value*xbar) if @x.value>0
    end

    def arguments
      (@x.arguments + @y.arguments).uniq
    end

  end

end
