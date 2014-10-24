# Autodiff

TODO: Write a gem description

## Installation

Add this line to your application's Gemfile:

    gem 'autodiff'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install autodiff

## Usage


Factory ideas

AD.x <- new variable
AD.M <- new matrix
AD.v <- new vector
AD.c <- constant
AD.C <- matrix valued constant

rest done via overloading

t is a term
t.arguments - lists all variables for a term
t.set - set arguments directly
t.gradient - t.arguments.map(&:gradient)

AD.apply(M, term)

## Contributing

1. Fork it ( https://github.com/[my-github-username]/autodiff/fork )
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request
