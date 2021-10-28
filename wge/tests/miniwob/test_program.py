import abc
import copy
import pytest

from wge.miniwob.fields import Fields
from wge.miniwob.state import MiniWoBState, DOMElement
from wge.miniwob.program import *
from wge.utils import word_tokenize


class ProgramTester(object):
    @pytest.fixture
    def fields(self):
        return Fields({"key": "value", "key2": "value2"})

    @pytest.fixture
    def env(self, dom, fields):
        state = MiniWoBState(
            "This is the Utterance.", fields, dom)
        env = ExecutionEnvironment(state)
        return env

    @pytest.fixture
    def dom(self):
        dom = {
            str("top"): 0, str("height"): 210, str("width"): 500,
            str("tag"): str("BODY"), str("ref"): 30,
            str("children"):
              [{str("top"): 0, str("height"): 210,
                str("width"): 160, str("tag"): str("DIV"),
                str("ref"): 31,
                str("children"):
                  [{str("top"): 50, str("height"): 4,
                    str("width"): 160, str("tag"): str("DIV"),
                    str("ref"): 32, str("children"):
                      [{str("text"): str("ONE."), str("top"): 165,
                        str("height"): 40, str("width"): 40,
                        str("tag"): str("BUTTON"), str("ref"): 33,
                        str("children"): [],
                        str("left"): 5},
                       {str("text"): str("TWO;"),
                        str("top"): 135, str("height"): 40,
                        str("width"): 40,
                        str("tag"): str("BUTTON"),
                        str("ref"): 34, str("children"): [],
                        str("left"): 95},
                       {str("text"): str("text"),
                        str("top"): 0, str("height"): 0,
                        str("width"): 0, str("tag"): str("t"),
                        str("children"): [], str("left"): 0},
                       {str("text"): str(""), str("top"): 151,
                        str("value"): str(""), str("height"): 19,
                        str("width"): 140,
                        str("tag"): str("INPUT_text"),
                        str("ref"): 127, str("children"): [],
                        str("left"): 2},
                       {str("text"): str(""), str("top"): 161,
                        str("value"): str(""), str("height"): 19,
                        str("width"): 140,
                        str("tag"): str("INPUT_checkbox"),
                        str("ref"): 128, str("children"): [],
                        str("left"): 2},
                       {str("text"): str(""), str("top"): 141,
                        str("value"): str(""), str("height"): 19,
                        str("width"): 140,
                        str("tag"): str("INPUT_radio"),
                        str("ref"): 129, str("children"): [],
                        str("left"): 2},
                       {str("text"): str("1 2 3 4 5 6"),
                        str("top"): 141,
                        str("value"): str(""), str("height"): 19,
                        str("width"): 140,
                        str("tag"): str("title"),
                        str("ref"): 130, str("children"): [],
                        str("left"): 2}
                      ],
                    str("left"): 0}
                  ],
                str("left"): 0}],
            str("left"): 0}
        return dom

    @pytest.fixture
    def dom_elem(self, dom):
        state = MiniWoBState("", None, dom)
        return state.dom_elements[-1]


class TestExecutionEnv(ProgramTester):
    def test_utterance(self, env, dom, dom_elem):
        # Test that the utterance is correct.
        assert env.tokens == word_tokenize("This is the Utterance.")

        # Update utterance.
        env.set_last(dom_elem)
        new_state = MiniWoBState("new utterance", None, dom)
        env.observe(new_state)

        # Test new value
        assert env.tokens == word_tokenize("new utterance")

    def test_buttons(self, env, dom, dom_elem):
        # Test that buttons are reported
        buttons = env.buttons
        assert len(buttons) == 2
        text = sorted([buttons[0].text, buttons[1].text])
        assert text == ["ONE.", "TWO;"]

        # Make a new observation
        env.set_last(dom_elem)
        new_dom = copy.deepcopy(dom)
        new_dom["children"][0]["children"][0]["children"][0]["text"] = "Hello"
        new_state = MiniWoBState("new utterance", None, new_dom)
        env.observe(new_state)

        # Make sure changes are reflected
        buttons = env.buttons
        assert len(buttons) == 2
        text = sorted([buttons[0].text, buttons[1].text])
        assert text == ["Hello", "TWO;"]

    def test_text(self, env, dom, dom_elem):
        # Test that text elements are reported
        text = env.text
        assert len(text) == 1
        assert text[0].text == "text"

        # Make a new observation
        env.set_last(dom_elem)
        new_dom = copy.deepcopy(dom)
        new_dom["children"][0]["children"][0]["children"][2]["text"] = "Hello"
        new_state = MiniWoBState("new utterance", None, new_dom)
        env.observe(new_state)

        # Make sure changes are reflected
        text = env.text
        assert len(text) == 1
        assert text[0].text == "Hello"

    def test_input(self, env, dom, dom_elem):
        # Test that env recognizes all types of input elems
        input_elems = env.input
        assert len(input_elems) == 3
        assert input_elems[0].text == ""
        assert input_elems[1].text == ""
        assert input_elems[2].text == ""

        # Make a new observation
        env.set_last(dom_elem)
        new_dom = copy.deepcopy(dom)
        new_dom["children"][0]["children"][0]["children"][3]["text"] = "Hello"
        new_state = MiniWoBState("new utterance", None, new_dom)
        env.observe(new_state)

        # Make sure changes are reflected
        input_elems = env.input
        assert len(input_elems) == 3
        text = "".join(
            [input_elems[0].text, input_elems[1].text, input_elems[2].text])
        assert text == "Hello"

    def test_last(self, env, dom, dom_elem):
        # Last should not be set yet.
        with pytest.raises(Exception):
            env.last

        # Set last and check
        env.set_last(dom_elem)
        assert env.last == dom_elem

        # Check can't double update last
        with pytest.raises(Exception):
            env.set_last(dom_elem)

        # Set a new observation
        state = MiniWoBState("utt", None, dom)

        # Check can't double update observation
        env.observe(state)
        with pytest.raises(Exception):
            env.observe(state)

    def test_valid_strings(self, env, dom, dom_elem):
        valid_strings = set(
            ["1 2 3", "2 3 4", "3 4 5", "4 5 6",
             "1 2", "2 3", "3 4", "4 5", "5 6",
             "1", "2", "3", "4", "5", "6", "ONE", "TWO", "text"])
        assert valid_strings == env.valid_strings

        # Make a new observation
        env.set_last(dom_elem)
        new_dom = copy.deepcopy(dom)
        new_dom["children"][0]["children"][0]["children"][3]["text"] = \
                "A B C D"
        new_state = MiniWoBState("new utterance", None, new_dom)
        env.observe(new_state)

        valid_strings.add("A B C")
        valid_strings.add("B C D")
        valid_strings.add("A B")
        valid_strings.add("B C")
        valid_strings.add("C D")
        valid_strings.add("A")
        valid_strings.add("B")
        valid_strings.add("C")
        valid_strings.add("D")

        # Check updated valid_strings
        assert valid_strings == env.valid_strings


class MockReturnElementSet(ProgramToken):
    """Executes to the element it is constructed around."""
    def __init__(self, element):
        self._elem = element

    def execute(self, env):
        return ElementSet([self._elem])

    @property
    def return_type(self):
        return ElementSet


class TokenTester(ProgramTester):
    @abc.abstractmethod
    def test_execute(self, env, fields, dom, dom_elem):
        raise NotImplementedError()


class ActionTokenTester(TokenTester):
    @abc.abstractmethod
    def test_consistent(self, env, dom, dom_elem):
        raise NotImplementedError()


class TestClickToken(ActionTokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        button = env.elements[0]
        click = ClickToken(MockReturnElementSet(button))
        result = click.execute(env)

        # Type check
        assert isinstance(result, click.return_type)
        assert isinstance(result, MiniWoBElementClick)

        # Check that click is on the button
        assert result.ref == button.ref

        # Check that last is updated
        assert env.last == button

    def test_consistent(self, env, dom, dom_elem):
        button = env.elements[0]
        click = ClickToken(MockReturnElementSet(button))

        wob_click = MiniWoBElementClick(button)
        inconsistent_click = MiniWoBElementClick(env.elements[1])

        assert click.consistent(env, wob_click)
        assert not click.consistent(env, inconsistent_click)

        click = ClickToken(ButtonsToken())
        button_click1 = MiniWoBElementClick(env.buttons[0])
        button_click2 = MiniWoBElementClick(env.buttons[1])

        assert click.consistent(env, button_click1)
        assert click.consistent(env, button_click2)
        assert not click.consistent(env, MiniWoBElementClick(env.input[0]))

        # Not consistent with other types of actions
        assert not click.consistent(
                env, MiniWoBFocusAndType(env.buttons[0], ""))
        assert not click.consistent(env, MiniWoBType(""))


class TestTypeToken(ActionTokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        type_token = TypeToken(StringToken("Hello, world!"))
        type_action = type_token.execute(env)

        assert isinstance(type_action, type_token.return_type)
        assert isinstance(type_action, MiniWoBType)

        assert type_action.text == "Hello, world!"

    def test_consistent(self, env, dom, dom_elem):
        type_token = TypeToken(StringToken("Hello, world!"))

        assert type_token.consistent(env, MiniWoBType("Hello, world!"))
        # Test compatibility with non-unicode
        assert type_token.consistent(env, MiniWoBType("Hello, world!"))
        # Check is sensitive to punctuation
        assert not type_token.consistent(env, MiniWoBType("Hello, world"))
        # Check is sensitive to capitalization
        assert not type_token.consistent(env, MiniWoBType("hello, world!"))

        type_token = TypeToken(UtteranceSelectorToken(0, 2))

        # Test that it works with utterance selection
        assert type_token.consistent(env, MiniWoBType("This is"))

        # Not consistent with other types of actions
        assert not type_token.consistent(
                env, MiniWoBFocusAndType(env.buttons[0], "This is"))
        assert not type_token.consistent(
                env, MiniWoBElementClick(env.buttons[0]))


class TestFocusAndTypeToken(ActionTokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        for elem in env.elements:
            if elem.tag == "input_text":
                text_elem = elem

        text_returner = MockReturnElementSet(text_elem)
        type_token = FocusAndTypeToken(
                text_returner, StringToken("Hello, world!"))
        type_action = type_token.execute(env)

        assert isinstance(type_action, type_token.return_type)
        assert isinstance(type_action, MiniWoBFocusAndType)

        assert type_action.ref == text_elem.ref
        assert type_action.text == "Hello, world!"

    def test_consistent(self, env, dom, dom_elem):
        for elem in env.elements:
            if elem.tag == "input_text":
                text_elem = elem

        text_returner = MockReturnElementSet(text_elem)
        type_token = FocusAndTypeToken(
                text_returner, StringToken("Hello, world!"))
        type_action = MiniWoBFocusAndType(text_elem, "Hello, world!")

        # Check that you are consistent
        assert type_token.consistent(env, type_action)

        # Same text diff element
        type_action = MiniWoBFocusAndType(env.elements[0], "Hello, world!")
        assert not type_token.consistent(env, type_action)

        # Same element diff text
        type_action = MiniWoBFocusAndType(text_elem, "Hello, world")
        assert not type_token.consistent(env, type_action)

        type_action = MiniWoBFocusAndType(text_elem, "hello, world!")
        assert not type_token.consistent(env, type_action)


class TestStringToken(TokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # Test empty string
        s = StringToken("")
        result = s.execute(env)

        assert isinstance(result, s.return_type)
        assert isinstance(result, str)
        assert result == ""

        # Test non-empty string
        s = StringToken("Hello, world!")
        result = s.execute(env)

        assert isinstance(result, s.return_type)
        assert isinstance(result, str)
        assert result == "Hello, world!"

        # Test non-unicode
        with pytest.raises(Exception):
            s = StringToken("Hello, world!")


class TestFieldsValueSelectorToken(TokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # Negative out of bounds
        with pytest.raises(Exception):
            selector = FieldsValueSelectorToken(-1)
            selector.execute(env)

        # Positive out of bounds
        with pytest.raises(Exception):
            selector = FieldsValueSelectorToken(2)
            selector.execute(env)

        selector = FieldsValueSelectorToken(0)
        result = selector.execute(env)

        assert isinstance(result, selector.return_type)
        assert isinstance(result, str)
        assert result == "value"

        selector = FieldsValueSelectorToken(1)
        result = selector.execute(env)

        assert isinstance(result, selector.return_type)
        assert isinstance(result, str)
        assert result == "value2"


class TestUtteranceSelectorToken(TokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # Start and end out of bounds
        with pytest.raises(Exception):
            selector = UtteranceSelectorToken(5, 6)
            selector.execute(env)

        # End out of bounds
        with pytest.raises(Exception):
            selector = UtteranceSelectorToken(1, 6)
            selector.execute(env)

        # Start bigger than end
        with pytest.raises(Exception):
            selector = UtteranceSelectorToken(4, 3)
            selector.execute(env)

        # Negative indices
        with pytest.raises(Exception):
            selector = UtteranceSelectorToken(4, -1)
            selector.execute(env)

        with pytest.raises(Exception):
            selector = UtteranceSelectorToken(-2, 5)
            selector.execute(env)

        # Pass both indices
        selector = UtteranceSelectorToken(2, 4)
        result = selector.execute(env)

        assert isinstance(result, selector.return_type)
        assert isinstance(result, str)
        assert result == "the Utterance"

        # Length 1
        selector = UtteranceSelectorToken(0, 1)
        result = selector.execute(env)

        assert isinstance(result, selector.return_type)
        assert isinstance(result, str)
        assert result == "This"

        # Check funky period spacing
        selector = UtteranceSelectorToken(3, 5)
        result = selector.execute(env)

        assert isinstance(result, selector.return_type)
        assert isinstance(result, str)
        assert result == "Utterance."


class TestSingleElementsSelectorToken(TokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # Test input
        input_token = InputElementsToken()
        result = input_token.execute(env)
        assert env.input == result
        assert isinstance(result, input_token.return_type)
        assert isinstance(result, ElementSet)

        # Test buttons
        buttons_token = ButtonsToken()
        result = buttons_token.execute(env)
        assert env.buttons == result
        assert isinstance(result, buttons_token.return_type)
        assert isinstance(result, ElementSet)

        # Test text
        text_token = TextToken()
        result = text_token.execute(env)
        assert env.text == result
        assert isinstance(result, text_token.return_type)
        assert isinstance(result, ElementSet)

        # Reset DOM
        env.set_last(dom_elem)
        new_dom = {
            str("top"): 0, str("height"): 210, str("width"): 500,
            str("tag"): str("BODY"), str("ref"): 30,
            str("children"): [],
            str("left"): 0
        }
        new_state = MiniWoBState("new utterance", None, new_dom)
        env.observe(new_state)

        # Test input
        input_token = InputElementsToken()
        result = input_token.execute(env)
        assert env.input == result
        assert isinstance(result, input_token.return_type)
        assert isinstance(result, ElementSet)

        # Test buttons
        buttons_token = ButtonsToken()
        result = buttons_token.execute(env)
        assert env.buttons == result
        assert isinstance(result, buttons_token.return_type)
        assert isinstance(result, ElementSet)

        # Test text
        text_token = TextToken()
        result = text_token.execute(env)
        assert env.text == result
        assert isinstance(result, text_token.return_type)
        assert isinstance(result, ElementSet)

        # Test last
        last_token = LastToken()
        result = last_token.execute(env)
        assert ElementSet([dom_elem]) == result
        assert isinstance(result, last_token.return_type)
        assert isinstance(result, ElementSet)


class DistanceTokenTester(TokenTester):
    @pytest.fixture
    def dom(self):
        dom = {
            str("top"): 0, str("height"): 0, str("width"): 0,
            str("tag"): str("BODY"), str("ref"): 0,
            str("left"): 0, str("children"): [
                {
                    str("top"): 100, str("left"): 100,
                    str("height"): 5, str("width"): 5,
                    str("ref"): 1, str("tag"): str("BUTTON"),
                    str("text"): str("ONE"), str("children"): []
                },
                {
                    str("top"): 0, str("left"): 210,
                    str("height"): 5, str("width"): 5,
                    str("ref"): 2, str("tag"): str("BUTTON"),
                    str("text"): str("TWO"), str("children"): []
                },
                {
                    str("top"): 0, str("left"): 0,
                    str("height"): 5, str("width"): 5,
                    str("ref"): 3, str("tag"): str("BUTTON"),
                    str("text"): str("THREE"), str("children"): []
                }
            ]
        }
        return dom


class TestSameRowToken(DistanceTokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # TODO: Refactor these tests (put the loop into own method)
        # Test 1 same row
        # (y, height) pairs
        included = [
            (101, 5),
            (99, 5),
            (100, 0),
            (105, 0),
            (99, 1),
            (80, 40),
            (101, 1)
        ]

        for (top, height) in included:
            new_dom = copy.deepcopy(dom)
            new_dom["children"][1]["top"] = top
            new_dom["children"][1]["height"] = height
            new_state = MiniWoBState("utt", None, new_dom)
            same_row = SameRowToken(
                    MockReturnElementSet(new_state.dom.children[0]))
            dom_elem = env.buttons[0]
            env.set_last(dom_elem)
            env.observe(new_state)
            result = same_row.execute(env)

            assert isinstance(result, same_row.return_type)
            assert isinstance(result, ElementSet)
            assert len(result) == 1
            assert set([result[0].ref]) == set([2])

        # Test that these points aren't included
        excluded = [
            (99, 0), (98, 1), (79, 20), (106, 0), (106, 100)
        ]

        for (top, height) in excluded:
            new_dom = copy.deepcopy(dom)
            new_dom["children"][1]["top"] = top
            new_dom["children"][1]["height"] = height
            new_state = MiniWoBState("utt", None, new_dom)
            same_row = SameRowToken(
                    MockReturnElementSet(new_state.dom.children[0]))
            dom_elem = env.buttons[0]
            env.set_last(dom_elem)
            env.observe(new_state)
            result = same_row.execute(env)

            assert isinstance(result, same_row.return_type)
            assert isinstance(result, ElementSet)
            assert len(result) == 0

        # Test that you can have multiple neighbors
        new_dom = copy.deepcopy(dom)
        new_dom["children"][1]["top"] = 103
        new_dom["children"][1]["height"] = 10
        new_dom["children"][2]["top"] = 99
        new_dom["children"][2]["height"] = 10

        new_state = MiniWoBState("utt", None, new_dom)
        same_row = SameRowToken(
                MockReturnElementSet(new_state.dom.children[0]))
        dom_elem = env.buttons[0]
        env.set_last(dom_elem)
        env.observe(new_state)
        result = same_row.execute(env)

        assert isinstance(result, same_row.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 2


# TODO: Refactor these tests to be same as SameRow
class TestSameColToken(DistanceTokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # Test 1 same col
        # (left, width) pairs
        included = [
            (101, 5),
            (99, 5),
            (100, 0),
            (105, 0),
            (99, 1),
            (80, 40),
            (101, 1)
        ]

        for (left, width) in included:
            new_dom = copy.deepcopy(dom)
            new_dom["children"][1]["left"] = left
            new_dom["children"][1]["width"] = width
            new_state = MiniWoBState("utt", None, new_dom)
            same_col = SameColToken(
                    MockReturnElementSet(new_state.dom.children[0]))
            dom_elem = env.buttons[0]
            env.set_last(dom_elem)
            env.observe(new_state)
            result = same_col.execute(env)

            assert isinstance(result, same_col.return_type)
            assert isinstance(result, ElementSet)
            assert len(result) == 1
            assert set([result[0].ref]) == set([2])

        # Test that these points aren't included
        excluded = [
            (99, 0), (98, 1), (79, 20), (106, 0), (106, 100)
        ]

        for (left, width) in excluded:
            new_dom = copy.deepcopy(dom)
            new_dom["children"][1]["left"] = left
            new_dom["children"][1]["width"] = width
            new_state = MiniWoBState("utt", None, new_dom)
            same_col = SameColToken(
                    MockReturnElementSet(new_state.dom.children[0]))
            dom_elem = env.buttons[0]
            env.set_last(dom_elem)
            env.observe(new_state)
            result = same_col.execute(env)

            assert isinstance(result, same_col.return_type)
            assert isinstance(result, ElementSet)
            assert len(result) == 0

        # Test that you can have multiple neighbors
        new_dom = copy.deepcopy(dom)
        new_dom["children"][1]["left"] = 103
        new_dom["children"][1]["width"] = 10
        new_dom["children"][2]["left"] = 99
        new_dom["children"][2]["width"] = 10

        new_state = MiniWoBState("utt", None, new_dom)
        same_col = SameColToken(
                MockReturnElementSet(new_state.dom.children[0]))
        dom_elem = env.buttons[0]
        env.set_last(dom_elem)
        env.observe(new_state)
        result = same_col.execute(env)

        assert isinstance(result, same_col.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 2


class TestNearToken(TokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # Three button simple DOM
        dom = {
            str("top"): 0, str("height"): 0, str("width"): 0,
            str("tag"): str("BODY"), str("ref"): 0,
            str("left"): 0, str("children"): [
                {
                    str("top"): 100, str("left"): 100,
                    str("height"): 5, str("width"): 5,
                    str("ref"): 1, str("tag"): str("BUTTON"),
                    str("text"): str("ONE"), str("children"): []
                },
                {
                    str("top"): 0, str("left"): 210,
                    str("height"): 5, str("width"): 5,
                    str("ref"): 2, str("tag"): str("BUTTON"),
                    str("text"): str("TWO"), str("children"): []
                },
                {
                    str("top"): 0, str("left"): 0,
                    str("height"): 5, str("width"): 5,
                    str("ref"): 3, str("tag"): str("BUTTON"),
                    str("text"): str("THREE"), str("children"): []
                }
            ]
        }
        env = ExecutionEnvironment(MiniWoBState("utt", None, dom))
        dom_elem = env.buttons[0]

        # Check all the pts that are near ONE
        included_points = [
            (75, 95), (75, 105),    # Edge on left
            (95, 125), (105, 125),  # Edge on bottom
            (125, 105), (125, 95),  # Edge on right
            (95, 75), (105, 75),    # Edge on top
            (100, 100),             # Full overlap
            (95, 95), (105, 95),    # Overlap top corners
            (95, 105), (105, 105),  # Overlap bottom corners
            (81, 81),               # Top left corner
            (119, 81),              # Top right corner
            (119, 119),             # Bottom right corner
            (81, 119),              # Bottom left corner
        ]

        # Check that all of the included pts are there
        for (left, top) in included_points:
            new_dom = copy.deepcopy(dom)
            new_dom["children"][1]["left"] = left
            new_dom["children"][1]["top"] = top
            new_state = MiniWoBState("utt", None, new_dom)
            near = NearToken(MockReturnElementSet(new_state.dom.children[0]))
            env.set_last(dom_elem)
            env.observe(new_state)
            dom_elem = env.buttons[0]
            result = near.execute(env)

            assert isinstance(result, near.return_type)
            assert isinstance(result, ElementSet)
            assert len(result) == 1
            assert set([result[0].ref]) == set([2])

        excluded_points = [
            (74, 95), (75, 106),    # Edge on left
            (94, 125), (105, 126),  # Edge on bottom
            (125, 106), (126, 95),  # Edge on right
            (95, 74), (106, 75),    # Edge on top
            (0, 0),                 # Not even close
            (80, 81),               # Top left corner
            (119, 80),              # Top right corner
            (120, 119),             # Bottom right corner
            (81, 120),              # Bottom left corner
        ]

        # Check that all of the excluded pts are not there
        for (left, top) in excluded_points:
            new_dom = copy.deepcopy(dom)
            new_dom["children"][1]["left"] = left
            new_dom["children"][1]["top"] = top
            new_state = MiniWoBState("utt", None, new_dom)
            near = NearToken(MockReturnElementSet(new_state.dom.children[0]))
            env.set_last(dom_elem)
            env.observe(new_state)
            dom_elem = env.buttons[0]
            result = near.execute(env)

            assert isinstance(result, near.return_type)
            assert isinstance(result, ElementSet)
            assert len(result) == 0

        # Check that you can have multiple pts in there
        new_dom = copy.deepcopy(dom)
        left, top = included_points[0]
        new_dom["children"][1]["left"] = left
        new_dom["children"][1]["top"] = top
        left, top = included_points[1]
        new_dom["children"][2]["left"] = left
        new_dom["children"][2]["top"] = top
        new_state = MiniWoBState("utt", None, new_dom)
        near = NearToken(MockReturnElementSet(new_state.dom.children[0]))
        env.set_last(dom_elem)
        env.observe(new_state)

        result = near.execute(env)
        assert isinstance(result, near.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 2
        assert set([result[0].ref, result[1].ref]) == set([2, 3])

        # Check for more than one input element
        near = NearToken(ButtonsToken())

        result = near.execute(env)
        assert isinstance(result, near.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 3
        assert set(
            [result[0].ref, result[1].ref, result[2].ref]) == set([1, 2, 3])


class TestLikeToken(TokenTester):
    def test_execute(self, env, fields, dom, dom_elem):
        # Test from StringToken
        like = LikeToken(StringToken("ONE"))
        result = like.execute(env)

        assert isinstance(result, like.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 1
        assert result[0].text == "ONE."

        # Test from UtteranceSelectorToken
        env.set_last(dom_elem)
        new_state = MiniWoBState("TWO utterance blah", None, dom)
        env.observe(new_state)

        like = LikeToken(UtteranceSelectorToken(0, 1))
        result = like.execute(env)

        assert isinstance(result, like.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 1
        assert result[0].text == "TWO;"

        # Test return none
        like = LikeToken(StringToken("abcd"))
        result = like.execute(env)

        assert isinstance(result, like.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 0

        # Test punctuation and whitespace
        like = LikeToken(StringToken("T ?W O ."))
        result = like.execute(env)

        assert isinstance(result, like.return_type)
        assert isinstance(result, ElementSet)
        assert len(result) == 1
        assert result[0].text == "TWO;"
