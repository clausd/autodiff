require 'java'
require './jar/mallet-deps.jar'
require './jar/mallet.jar'

module Autodiff
  module Mallet
    class Simple
      include Java::cc::mallet::optimize::Optimizable::ByGradientValue

      attr_accessor :expression

      def initialize(expression)
        @expression = expression
      end

      def getValue
        @expression.value
      end

      def getValueGradient(gradient)
        expression.accumulate(1)
        j = 0
        @expression.arguments.each do |v|
          a = v.gradient_array
          # jruby can't assign to range
          (j..(j+a.size-1)).each_with_index {|k,i| gradient[k] = a[i]}
          j += a.size
        end
      end

      # The following get/set methods satisfy the Optimizable interface

      def getNumParameters
        @expression.arguments.map {|a| a.value_array.size }.inject(&:+)
      end

      def getParameter(i)
        # EXPENSIVE
        # TODO fix this
        a = []
        getParameters(a)
        a[i]
      end

      def getParameters(buffer)
        j = 0
        @expression.arguments.each do |v|
          a = v.value_array
          # jruby can't assign to range
          (j..(j+a.size-1)).each_with_index {|k,i| buffer[k] = a[i]}
          j += a.size
        end
      end

      def setParameter(i, r)
        # EXPENSIVE
        # TODO fix this
        a = []
        getParameters(a)
        a[i] = r
        setParameters(a)
      end

      def setParameters(newParameters)
        j = 0
        @expression.arguments.each do |v|
          a = v.value_array
          v.set(newParameters[j..(j+a.size-1)])
          j += a.size
        end
      end

      def solve
        construct = Java::cc::mallet::optimize::LimitedMemoryBFGS.java_class.constructor(Java::cc::mallet::optimize::Optimizable::ByGradientValue)
        optimizer = construct.new_instance(self)
        optimize_method = optimizer.java_class.declared_method(:optimize)
        optimize_method.invoke(optimizer)
      end

    end
  end
end
