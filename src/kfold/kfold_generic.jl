# Define the base function
function kfold(x::Any, args...)
    # Default behavior
    println("Default behavior")
end

# Define a method for a specific type
function kfold(x::Array, args...)
    # Behavior for arrays
    println("Array behavior")
end

# Define a method for another specific type
function kfold(x::String, args...)
    # Behavior for strings
    println("String behavior")
end

# Define the is.kfold function
function is_kfold(x::Any)
    isa(x, kfold) && is_loo(x)
end

# Define the dim.kfold function
function dim_kfold(x::Any)
    size(x)
end
