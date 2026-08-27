from waffle.plotting import build_cube_figure


def test_cube_figure_places_single_point():
    fig = build_cube_figure(0.25, 0.5, 0.75)
    trace = fig.data[0]
    assert list(trace.x) == [0.25]
    assert list(trace.y) == [0.5]
    assert list(trace.z) == [0.75]
    assert trace.mode == "markers+text"
    scene = fig.layout.scene
    assert scene.xaxis.range == (0, 1)
    assert scene.yaxis.range == (0, 1)
    assert scene.zaxis.range == (0, 1)
