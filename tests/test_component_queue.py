import json

from tools.run_component_queue import prerequisite_state


def job(tmp_path):
    return dict(wait_for_status=str(tmp_path/'training.json'),
                producer_queue_status=str(tmp_path/'queue.json'), producer_job='graph_fm')


def test_do_not_consume_uncommitted_checkpoint(tmp_path):
    item=job(tmp_path)
    (tmp_path/'training.json').write_text(json.dumps(dict(phase='training',update=2000)))
    assert prerequisite_state(item)[0]=='waiting'
    (tmp_path/'training.json').write_text(json.dumps(dict(phase='completed',update=2048)))
    assert prerequisite_state(item)==('ready',None)


def test_failed_and_disappeared_producers_are_not_restarted(tmp_path,monkeypatch):
    item=job(tmp_path)
    (tmp_path/'queue.json').write_text(json.dumps(dict(phase='running',pid=123,jobs={})))
    def missing(pid, signal):
        raise ProcessLookupError()
    monkeypatch.setattr('tools.run_component_queue.os.kill',missing)
    assert prerequisite_state(item)[0]=='failed'
    (tmp_path/'training.json').write_text(json.dumps(dict(phase='failed',error='bad labels')))
    assert prerequisite_state(item)==('failed','bad labels')


def test_terminal_queue_without_output_cannot_be_counted_complete(tmp_path):
    item=job(tmp_path)
    (tmp_path/'queue.json').write_text(json.dumps(dict(phase='completed',jobs={})))
    assert prerequisite_state(item)[0]=='failed'
    assert prerequisite_state({})==('ready',None)
