[Back to Overview](../../README.md)

# babilim.core.annotations

> A collection of helpful annotations.

This code is under the MIT License and requires the abc package.



A class decorator is a base class that is used for all annotations that should be usable with python classes.
Regular annotations will not work with classes.

This is a helper class that can be used when writing annotations.


---
---
## *class* **RunOnlyOnce**(_ClassDecorator)

A decorator that ensures a function in an object gets only called exactly once.

* **f**: The function that should be wrapped.


The run only once annotation is fundamental for the build function pattern, whereas it allows to write a function which is only called once, no matter how often it gets called. This behaviour is very usefull for creating variables on the GPU only once in the build and not on every run of the neural network.

> Important: This is for use with the build function in a module. Ensuring it only gets called once and does not eat memory on the gpu.

Using this in an example function which prints the parameter only yields on printout, even though the function gets called multiple times.

Example:
```python
@RunOnlyOnce
def test_fun(msg):
    print(msg)
    
test_fun("Foo")
test_fun("Foo")
test_fun("Foo")
test_fun("Foo")
test_fun("Foo")

```
Output:
```
Foo

```

# Jupyter Notebook Helpers

---
### *def* **extend_class**(clazz, function_name)

Extend a class by the function decorated with this decorator.

* **clazz**: The class that should be decorated.
* **function_name**: The name that the function in the class should have. (Can be different than unbound name of the class.)


This annotation can be used for developing code in a jupyter notebook. It allows you to define a class in separate cells, like the following example. This gives you the exploratory capabilities of jupyter while developing a class that can be later exported (using nbdev) and used in production.

Example:
```python
# first cell
class TestClass(object):
    def __init__(self, var):
        self.my_var = var
        
# later cell
test = TestClass(42)
        
# again later cell
@extend_class(TestClass, "foo")
def __foo(self, name):
    print("self.my_var={}".format(self.my_var))
    print("name={}".format(name))
    self.name = name

# and again later cell
test.foo(name="Hello")
print(test.name)
```
Output:
```
self.my_var=42
name=Hello
Hello

```

