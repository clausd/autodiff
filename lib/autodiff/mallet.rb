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
        @expression.arguments.each_with_index {|v,i| gradient[i] = v.gradient}
      end

      # The following get/set methods satisfy the Optimizable interface

      def getNumParameters
        @expression.arguments.count
      end

      def getParameter(i)
        @expression.arguments[i].value
      end

      def getParameters(buffer)
        @expression.arguments.each_with_index {|v,i| buffer[i] = v.value}
      end

      def setParameter(i, r)
        @expression.arguments[i].set(r)
      end

      def setParameters(newParameters)
        @expression.arguments.each_with_index {|v,i| v.set(newParameters[i])}
      end

      def solve(initial)
        self.setParameters(initial)
        construct = Java::cc::mallet::optimize::LimitedMemoryBFGS.java_class.constructor(Java::cc::mallet::optimize::Optimizable::ByGradientValue)
        optimizer = construct.new_instance(self)
        optimize_method = optimizer.java_class.declared_method(:optimize)
        optimize_method.invoke(optimizer)
      end

    end
  end
end
