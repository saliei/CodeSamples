import re

class Parser:
    """A class to parase the parameter file.
    
    This class will parse the parameter file, provided that it
    follows the general template provide, with `[General]` section
    being the first one, and the `[Planet]` sections following that.

    Attributes
    ----------
    filename: str
        the file name for the parameter file.

    Methods
    -------
    parse():
        function for parsing the file.

    """
    def __init__(self, filename: str) -> None:
        """Constructor for Parser class.
        
        Parameters
        ----------
        filename: str
            The file name for parameter file.
        """
        self.filename = filename
        try:
            with open(self.filename, 'r') as param_file:
                self.param_content = param_file.read()
        except IOError:
            print("The initial parameter file can't be accessed.")
        self.global_vars, self.planets = self.parse()
        
    def parse(self) -> tuple[dict, dict]:
        """Fuction to do the actual parsing of the parameter file.

        This function will first split the parameter file into `[General]`
        and `[Planet]` section, then it will go through each section and 
        extract the variables based on equality sign and convert the 
        values to float.

        Parameters
        ----------
        filename: str
            The file name for parameter file.

        Returns
        -------
        global_vars: dict
            A dictionary of global variables and values.
        planets: dict
            A dictionary of of dictionary with planet number 
                and it's various properties.

        """
        _global_vars = {}
        _planets = {}
        _pattern = re.compile(r"\[Global]\n|\[Planet]\n")
        _sections = _pattern.split(self.param_content)
        for _section in _sections:
            if not _section:
                _sections.remove(_section)
        _vars = _sections[0].strip().split("\n")
        for _var in _vars:
            _var_string = _var.split('=')[0].strip()
            _var_value = _var.split('=')[-1]
            _global_vars[_var_string] = float(_var_value)
        for _id, _section in enumerate(_sections[1:]):
            _planets[_id] = {}
            _vars = _section.strip().split("\n")
            for _var in _vars:
                _var_string = _var.split('=')[0].strip()
                if _var_string == "name":
                    _var_value = _var.split('=')[-1].strip()
                else:
                    _var_value = float(_var.split('=')[-1].strip())
                _planets[_id][_var_string] = _var_value

        return _global_vars, _planets


if __name__ == "__main__":
    parser = Parser("params.in")
    # sample use cases
    print("Print param file:")
    print(parser.param_content)

    print("Global Variables:")
    print(parser.global_vars)

    print("Planets:")
    print(parser.planets)

    print("Planet #1 mass:")
    print(parser.planets[0]["mass"])


