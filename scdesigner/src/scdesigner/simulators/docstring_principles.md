# Docstring Principles

- Writing docstrings: should follow the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).
  
  ```bash
  def example_function(param1, param2):
      """
      Brief summary of the function.
  
      A more detailed explanation of what the function does, including any
      important details or context.
  
      Parameters
      ----------
      param1 : int
          Description of the first parameter. Mention its purpose and any
          constraints or expected values.
      param2 : str
          Description of the second parameter. Include details about its
          usage or format.
  
      Returns
      -------
      bool
          Description of the return value. Explain what it represents and
          under what conditions it might vary.
  
      Raises
      ------
      ValueError
          Description of the error raised, including the circumstances
          under which it occurs.
  
      Examples
      --------
      >>> example_function(42, "hello")
      True
      """
      if not isinstance(param1, int):
          raise ValueError("param1 must be an integer.")
      if not isinstance(param2, str):
          raise ValueError("param2 must be a string.")
      return True
  ```
  
- Distinguish public and private APIs: make sure internal helper functions and private attributes are prefixed with `_`
  
- For internal functions, docstrings should be concise
  
- Write READMEs (A single root README or multiple READMEs for each component?)
  
  - Root README
    - Project overview and features
    - Installation (to be done after published online)
    - (Links to other guides)
  - Python package README: Python API overview
  - R package README: R API overview, python backend bridging
  - Examples README: List of examples and descriptions
- Provide one example for each class including as many public methods as possible. No need to write examples for every method.