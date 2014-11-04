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
        expression.partials.each_with_index {|v,i| gradient[i] = v}
      end

      # The following get/set methods satisfy the Optimizable interface

      def getNumParameters
        @expression.size
      end

      def getParameter(i)
        expression.scalars[i]
      end

      def getParameters(buffer)
        expression.scalars.each_with_index {|v,i| buffer[i] = v}
      end

      def setParameter(i, r)
        expression.arguments.each do |arg|
          if arg.position.first <= i && i <= arg.position.last
            expression.scalars[i] = r
            arg.set(expression.scalars[arg.position])
          end
        end
      end

      def setParameters(newParameters)
        expression.arguments.each do |arg|
          arg.set(newParameters[arg.position])
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
